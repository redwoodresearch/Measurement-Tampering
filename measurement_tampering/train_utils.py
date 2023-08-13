from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, Optional, TypedDict

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from measurement_tampering.model_with_answer_pred import ModelWithAnswerPred


class BatchData(TypedDict):
    passes: torch.Tensor  # bool (batch, nb_sensors)
    all_passes: torch.Tensor  # bool (batch)
    is_correct: torch.Tensor  # bool (batch)
    is_clean: torch.Tensor  # bool (batch)
    input_ids: torch.Tensor  # int (batch, seq_len)
    attention_mask: torch.Tensor  # bool (batch, seq_len)

    # sensor_locs: token indexes to read out each *individual sensor*
    sensor_locs: torch.Tensor  # long (batch, nb_sensors)

    # sensor_mask: which of the individual sensors is active (currently always true except for function correctness)
    sensor_mask: torch.Tensor  # long (batch, nb_sensors)

    # overall_loc: token index to read out the conjunction sensor and the ground truth sensor.
    # Assumes we want the same location for conjunction and ground truth
    overall_loc: torch.Tensor  # long (batch,) # loc for gt and all_passes

    counterfactual_tokens: torch.Tensor  # optional long (batch, nb_counterfacts, seq_len)
    ntp_mask: torch.Tensor  # optional bool (batch, seq_len)
    difficulty: torch.Tensor  # optional int (batch)
    swaps: torch.Tensor  # optional (batch_size,)
    overall_tamper_evidence: torch.Tensor  # optional bool (batch, ), used as ~tamper_detection junction_target if available
    individual_tamper_evidence: torch.Tensor  # optional bool (batch, nb_sensors)


def get_all_locs(data: BatchData):
    return torch.cat([data["overall_loc"][:, None], data["overall_loc"][:, None], data["sensor_locs"]], dim=-1)


Category = Literal["nano", "tamper", "clean", "positive_clean", "negative"]


# TODO: fix up elk dataset
class ElkDataset(Dataset, ABC):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> BatchData:
        ...

    @abstractmethod
    def filter_cat(self, cat: Category) -> "ElkDataset":
        ...

    @abstractmethod
    def exclude_half_negatives(self) -> "ElkDataset":
        ...

    @abstractmethod
    def add_counterfactuals(self, n_choices: int = 1) -> "ElkDataset":
        ...


def any_passes(data: BatchData) -> torch.Tensor:
    return (data["passes"] & data["sensor_mask"]).any(-1)


def is_consistent(batch_data: BatchData):
    return batch_data["all_passes"] | ~any_passes(batch_data)


def no_evidence_for_tamper(batch_data: BatchData):
    if "overall_tamper_evidence" in batch_data:
        return ~batch_data["overall_tamper_evidence"]
    else:
        return is_consistent(batch_data)


class TokenizedDataElkDataset(ElkDataset):
    def __init__(self, tokenized_data: BatchData):
        self.data = tokenized_data

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, idx) -> BatchData:
        return BatchData(**{k: v[idx] for k, v in self.data.items()})

    def filter(self, mask: torch.Tensor) -> TokenizedDataElkDataset:
        return TokenizedDataElkDataset(self[mask])

    def filter_cat(self, cat: Category) -> TokenizedDataElkDataset:
        if cat == "nano":
            return self.filter(self.data["is_correct"] & ~self.data["is_clean"])
        elif cat == "tamper":
            return self.filter(~self.data["is_correct"] & self.data["all_passes"])
        elif cat == "clean_or_neg":
            return self.filter(self.data["is_clean"] | ~self.data["all_passes"])
        elif cat == "clean_or_half_neg":
            return self.filter(self.data["is_clean"] | (~is_consistent(self.data)))
        elif cat == "clean_or_evidence_for_tamper":
            return self.filter(self.data["is_clean"] | ~no_evidence_for_tamper(self.data))
        elif cat == "abs_clean":
            return self.filter(self.data["is_clean"])
        elif cat == "positive_clean":
            return self.filter(self.data["is_correct"] & self.data["is_clean"])
        elif cat == "negative":
            return self.filter(~self.data["all_passes"])
        elif cat == "positive":
            return self.filter(self.data["all_passes"])
        elif cat == "nano_tamper":
            return self.filter(~self.data["is_clean"] & self.data["all_passes"])
        elif cat == "no_false_negatives":
            false_negative = self.data["is_correct"] & ~self.data["all_passes"]
            return self.filter(~false_negative)
        else:
            raise ValueError(f"Unknown category {cat}")

    def exclude_half_negatives(self) -> TokenizedDataElkDataset:
        return self.filter(is_consistent(self.data))

    def exclude_clean_tamper(self) -> TokenizedDataElkDataset:
        return self.filter(~(self.data["is_clean"] & self.data["all_passes"] & ~self.data["is_correct"]))

    def exclude_false_negatives(self) -> TokenizedDataElkDataset:
        return self.filter(~(self.data["is_correct"] & ~self.data["all_passes"]))

    def add_counterfactuals(self, n_choices: int = 1):
        out = self.data.copy()

        toks = out["input_ids"]
        r = [toks[torch.randperm(len(toks))] for _ in range(n_choices)]
        out["counterfactual_tokens"] = torch.stack(r, dim=1)

        return TokenizedDataElkDataset(out)


class LossAndPreds(NamedTuple):
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    sensor_values: torch.Tensor
    all_passes_value: torch.Tensor
    is_correct_value: torch.Tensor


def map_loss_and_preds(fn: Callable[[torch.Tensor], torch.Tensor], loss_and_preds: LossAndPreds) -> LossAndPreds:
    return LossAndPreds(
        loss=fn(loss_and_preds.loss),
        metrics={k: fn(v) for k, v in loss_and_preds.metrics.items()},
        sensor_values=fn(loss_and_preds.sensor_values),
        all_passes_value=fn(loss_and_preds.all_passes_value),
        is_correct_value=fn(loss_and_preds.is_correct_value),
    )


def loss_and_preds_to_device(loss_and_preds: LossAndPreds, device: torch.device) -> LossAndPreds:
    return map_loss_and_preds(lambda t: t.to(device), loss_and_preds)


TRUTH_SENSOR = 1
ALL_PASSES_SENSOR = 0


class Setting(ABC):
    @abstractmethod
    def create_get_losses(
        self, tokenizer, *args, **kwargs
    ) -> Callable[["ModelWithAnswerPred", BatchData], LossAndPreds]:
        """Return a function that computes the loss and predictions.

        Should always be called first."""
        ...

    @abstractmethod
    def load_data(self, data_dir_here: str, split: str, max_data: Optional[int] = None) -> ElkDataset:
        """Return train and val datasets"""
        ...

    @abstractmethod
    def initialization_tokens(self, tokenizer) -> tuple[int, int]:
        """Return the token ids for the sensor initialization tokens"""
        ...

    @abstractproperty
    def nb_individual_sensor_values(self) -> int:
        """Return the (max) number of individual sensor values on a single datapoint"""
        ...

    @property
    def nb_overall_sensor_values(self) -> int:
        """Return the max number of sensor values returned on one data point include conjunction and ground truth"""
        return self.nb_individual_sensor_values + 2

    @abstractproperty
    def val_splits(self) -> list[str]:
        """What are the names of the val splits."""
        ...

    @property
    def train_split(self) -> str:
        """What is the name of the main training split with answers"""
        return "train"

    @property
    def val_split(self) -> str:
        """What is the name of the main validation split with answers"""
        return self.val_splits[0]

    @property
    def answers_train_split(self) -> str:
        """What is the name of the coco training split, which should always have min_nb_answers answers."""
        return self.train_split

    @cached_property
    def other_sensors(self) -> list[int]:
        """What are the ids of the other sensors"""
        return list(range(2, self.nb_overall_sensor_values))

    @abstractproperty
    def seq_len(self) -> int:
        """What is the maximum sequence length"""
        ...


class BaseSimpleSetting(Setting):
    def single_tokenize(self, x: str, tokenizer):
        out = tokenizer(x)["input_ids"]
        assert len(out) == 1
        return out[0]

    def create_get_losses(
        self,
        tokenizer,
        ground_truth_weight: float = 0.0,
        token_loss_weight: float = 1.0,
        overall_loss_weight: float = 0.0,
        excl_sensor_mask: Optional[list[float]] = None,
        excluded_set: Literal["dirty", "dirty-positive"] = "dirty",
        junction_target: str = "all_passes",
        **kwargs,
    ) -> Callable[[ModelWithAnswerPred, BatchData], LossAndPreds]:
        pad_tok = self.single_tokenize("[PAD]", tokenizer)

        # @torch.compile
        def get_losses(model: ModelWithAnswerPred, batch: BatchData) -> LossAndPreds:
            if junction_target == "tamper_detection":
                batch["tamper_detection"] = no_evidence_for_tamper(batch)
            elif junction_target == "inconsistency_detection":
                batch["inconsistency_detection"] = is_consistent(batch)

            if model.bdas is not None:
                batch = model.bdas.generate_swaps_and_passes(batch)

            logits, sensor_logits = model(
                input_ids=batch["input_ids"][..., :-1],
                attention_mask=batch["attention_mask"][..., :-1],
                batch_data=batch,
            )

            # logit_answers_by_tok: (batch, seq, all tests - is correct - *each other test)

            labels = batch["input_ids"][..., 1:]
            token_loss_by_token = torch.nn.functional.cross_entropy(
                logits.view(labels.numel(), -1), labels.reshape(-1), reduction="none"
            ).view(labels.shape)
            next_token_mask = labels != pad_tok
            if "ntp_mask" in batch:
                next_token_mask = next_token_mask * batch["ntp_mask"][..., :-1]

            next_token_loss = (token_loss_by_token * next_token_mask).sum() / torch.clamp(next_token_mask.sum(), min=1)

            all_probes_actual = torch.cat(
                [batch[junction_target][..., None], batch["is_correct"][..., None], batch["passes"]], dim=-1
            )
            assert all_probes_actual.shape == sensor_logits.shape, (
                all_probes_actual.shape,
                sensor_logits.shape,
            )
            assert ~sensor_logits.isnan().any()
            assert sensor_logits.isfinite().all()
            probability_from_probes = torch.sigmoid(sensor_logits)
            all_probe_loss_by_loc = torch.nn.functional.binary_cross_entropy_with_logits(
                sensor_logits.reshape(-1),
                all_probes_actual.to(sensor_logits.dtype).reshape(-1),
                reduction="none",
            ).view(sensor_logits.shape)

            if excl_sensor_mask is not None:
                excl_sensor_mask_t = torch.tensor(excl_sensor_mask, device=sensor_logits.device, dtype=torch.bool)
                if excluded_set == "dirty":
                    excluded_mask = ~batch["is_clean"]
                elif excluded_set == "dirty-positive":
                    excluded_mask = ~batch["is_clean"] & batch["all_passes"]
                elif excluded_set == "dirty-positive-0":
                    excluded_mask = ~batch["is_clean"] & batch["passes"][..., 0]
                elif excluded_set == "dirty-positive-1":
                    excluded_mask = ~batch["is_clean"] & batch["passes"][..., 1]
                elif excluded_set == "dirty-positive-2":
                    excluded_mask = ~batch["is_clean"] & batch["passes"][..., 2]
                else:
                    raise ValueError(f"Unknown excluded set {excluded_set}")
                # half_neg_mask = ~batch["passes"].all(-1) & batch["passes"].any(-1)
                excl_ft_mask = ~excluded_mask[:, None] | excl_sensor_mask_t[None, :]

                all_probe_loss_by_loc *= excl_ft_mask

            assert {TRUTH_SENSOR, ALL_PASSES_SENSOR} == {0, 1}
            individual_sensor_losses_by_each = (all_probe_loss_by_loc[:, 2:] * batch["sensor_mask"]).mean(dim=0)
            individual_sensor_loss = individual_sensor_losses_by_each.sum(-1)
            conjunction_sensor_loss = all_probe_loss_by_loc[:, ALL_PASSES_SENSOR].mean()
            ground_truth_sensor_loss = all_probe_loss_by_loc[:, TRUTH_SENSOR].mean()

            total_other_weight = token_loss_weight + overall_loss_weight
            total_individual_sensor_weight = 1 - total_other_weight
            total_loss = (
                token_loss_weight * next_token_loss
                + total_individual_sensor_weight * individual_sensor_loss
                + overall_loss_weight
                * (conjunction_sensor_loss * (1 - ground_truth_weight) + ground_truth_sensor_loss * ground_truth_weight)
            )

            extra_metrics_highlighted_ntp = self.highlighted_ntp_metrics(next_token_mask, token_loss_by_token)

            return LossAndPreds(
                loss=total_loss,
                metrics={
                    "loss": total_loss,
                    "token_l": next_token_loss,
                    "individual_sensor_l": individual_sensor_loss,
                    **{f"sensor{i+1}_l": x for i, x in enumerate(individual_sensor_losses_by_each)},
                    "sensorall_l": conjunction_sensor_loss,
                    "sensorall_acc": ((probability_from_probes[:, 0] > 0.5) == batch[junction_target]).float().mean(),
                    "gt_l": ground_truth_sensor_loss,
                    "gt_acc": ((probability_from_probes[:, 1] > 0.5) == batch["is_correct"]).float().mean(),
                    **extra_metrics_highlighted_ntp,
                },
                sensor_values=sensor_logits[:, 2:],
                all_passes_value=sensor_logits[:, 0],
                is_correct_value=sensor_logits[:, 1],
            )

        return get_losses

    def highlighted_ntp_metrics(self, next_token_mask: torch.Tensor, token_loss_by_token: torch.Tensor):
        return {}


class TokenizedDataBaseSetting(BaseSimpleSetting):
    def load_data(self, data_dir_here: str, split: str, max_data: Optional[int] = None) -> TokenizedDataElkDataset:
        """Return train and data loaders"""

        return TokenizedDataElkDataset(BatchData(**slice_batch(torch.load(f"{data_dir_here}/{split}.pt"), max_data)))


def slice_batch(batch_data: BatchData, max_points: Optional[int] = None):
    return {k: v[:max_points] if max_points is not None else v for k, v in batch_data.items()}
