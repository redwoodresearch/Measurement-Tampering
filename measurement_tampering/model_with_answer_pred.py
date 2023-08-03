import contextlib
import json
import os
import random
from abc import ABC, abstractmethod, abstractproperty
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, TypeVar, Union

import attrs
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from diamonds.bdas import BoundlessDasLayer
from measurement_tampering.activations_utils import (
    find_call_lm_head,
    get_final_module,
    get_hidden_size,
    get_lm_head,
    get_module_by_layer,
    get_transformer,
)
from measurement_tampering.train_utils import ALL_PASSES_SENSOR, TRUTH_SENSOR, BatchData, get_all_locs

if TYPE_CHECKING:
    from measurement_tampering.activations_utils import GptModel


GetEmbedsMode = Literal["identity", "small_attn", "full_attn"]
GetSensorsMode = Literal["linear", "linear_shared"]


class WeightingAttn(torch.nn.Module):
    """single attn head which just weights some embeddings (identity OV)"""

    def __init__(self, n_embd, qk_dim: int = 16) -> None:
        super().__init__()
        self.q_mat = torch.nn.Linear(n_embd, qk_dim)
        self.k_mat = torch.nn.Linear(n_embd, qk_dim)
        self.qk_dim = qk_dim

    def probs(self, x):
        q_pos = torch.arange(x.shape[-2])[:, None].to(x.device)
        k_pos = torch.arange(x.shape[-2])[None, :].to(x.device)
        mask = q_pos >= k_pos
        scores = torch.where(
            mask[None],
            torch.einsum(
                "b q d, b k d, q k, -> b q k",
                self.q_mat(x),
                self.k_mat(x),
                mask,
                1 / torch.sqrt(torch.tensor(self.qk_dim).to(x)),
            ),
            torch.tensor(-10_000.0).to(x),
        )
        return torch.softmax(scores, dim=-1)

    def forward(self, x):
        return torch.einsum("b q k, b k v -> b q v", self.probs(x), x)


class SensorProbe(torch.nn.Module, ABC):
    @abstractproperty
    def weight(self) -> torch.Tensor:
        ...

    @abstractproperty
    def bias(self) -> torch.Tensor:
        ...


class LinearProbe(SensorProbe):
    def __init__(self, n_embd, probe_dim) -> None:
        super().__init__()
        self.probe = torch.nn.Linear(n_embd, probe_dim)

    def forward(self, x: torch.Tensor):
        return torch.einsum("b p d, p d -> b p", x, self.probe.weight) + self.probe.bias

    @property
    def weight(self):
        return self.probe.weight

    @property
    def bias(self):
        return self.probe.weight


class LinearShared(SensorProbe):
    def __init__(self, n_embd, probe_dim) -> None:
        super().__init__()
        self.probe = torch.nn.Linear(n_embd, 3)
        self.probe_dim = probe_dim
        assert {ALL_PASSES_SENSOR, TRUTH_SENSOR} == {0, 1}
        assert probe_dim >= 3

    def forward(self, x: torch.Tensor):
        return torch.einsum("b p d, p d -> b p", x, self.expanded_weight) + self.expanded_bias

    @property
    def expanded_weight(self):
        return torch.cat([self.probe.weight[:2]] + [self.probe.weight[None, -1]] * (self.probe_dim - 2), dim=0)

    @property
    def expanded_bias(self):
        return torch.cat([self.probe.bias[:2]] + [self.probe.bias[None, -1]] * (self.probe_dim - 2), dim=0)

    @property
    def weight(self):
        return self.probe.weight

    @property
    def bias(self):
        return self.probe.weight


@attrs.frozen
class GeneralMdRemoverArgs:
    use_labels: str = "all_passes"
    locked: bool = False


@attrs.frozen
class GeneralLayerMdRemoverArgs:
    layer: int
    remover_args: GeneralMdRemoverArgs


class MeanDiffRemover(torch.nn.Module):
    """Remove the difference of the means of the two clusters.

    Uses the eval time batch norm trick to compute the means of the two clusters.

    embedding_count is the number of different embeddings to save, can be seq_len in some cases

    Saves means *per each of these embedding_count many things* and expects all inputs to have shape (batch, embedding_count, n_embd)
    """

    def __init__(
        self, n_embd, remover_args: GeneralMdRemoverArgs = GeneralMdRemoverArgs(), embedding_count: int = 1023
    ) -> None:
        super().__init__()
        self.use_labels = remover_args.use_labels
        self.locked = remover_args.locked

        self.cum_sum: torch.Tensor
        self.cum_bias: torch.Tensor
        self.nbs: torch.Tensor
        self.register_buffer("cum_sum", torch.zeros((2, embedding_count, n_embd)))
        self.register_buffer("cum_bias", torch.zeros((embedding_count,)))
        self.register_buffer("nbs", torch.zeros((2,)))

    @property
    def mean0(self):
        return self.cum_sum[0] / torch.clamp(self.nbs[0], min=1)

    @property
    def mean1(self):
        return self.cum_sum[1] / torch.clamp(self.nbs[1], min=1)

    @property
    def bias(self):
        return self.cum_bias / self.nbs.sum()

    @property
    def amnestic_direction(self):
        delta = self.mean1 - self.mean0
        return delta / (torch.norm(delta, dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor, batch_data: BatchData):
        if self.cum_sum.device != x.device:
            print(f"Moving cum sum to device {x.device}")
            self.to(x.device)

        if self.training and not self.locked:
            assert batch_data is not None, "BatchData must be given at train time"
            assert x.ndim == 3, "Input must be of shape (batch, embedding_count/seq_len, features)"
            labels = batch_data[self.use_labels]

            if (labels == 0).any():
                self.cum_sum[0] += x[labels == 0].sum(0)  # sum over batch
                self.nbs[0] += (labels == 0).sum()
            if (labels == 1).any():
                self.cum_sum[1] += x[labels == 1].sum(0)
                self.nbs[1] += (labels == 1).sum()

            self.cum_bias += torch.einsum("bld,ld->l", x, self.amnestic_direction)

        if self.nbs[0].item() < 0.5 and self.nbs[1].item() < 0.5:
            print(f"WARNING: no labels seen yet, not removing mean diff: {self.nbs}")

        projected = project_per_position(x, self.amnestic_direction) + self.bias[:, None] * self.amnestic_direction

        return projected


class HookingMeanDiffRemover(torch.nn.Module):
    """
    layer None means last module
    """

    def __init__(self, remover: MeanDiffRemover, layer: int) -> None:
        super().__init__()
        self.remover = remover
        self.layer = layer

    # context manager to insert the hook which projects the activations
    @contextlib.contextmanager
    def hook_into(self, model: "GptModel", batch_data: Optional[BatchData] = None):
        # TODO: types don't make sense, I'm not sure how this is used
        target_module = get_module_by_layer(model, self.layer)

        def hook_fn(module, input, output):
            assert isinstance(output, tuple)
            x, *rest = output
            assert isinstance(x, torch.Tensor)
            x = self.remover(x, batch_data)
            return (x, *rest)

        handle = target_module.register_forward_hook(hook_fn)
        yield
        handle.remove()


class ModelWithAnswerPred(torch.nn.Module):
    model: "GptModel"
    get_embeds_for_sensors: torch.nn.Module
    sensor_md_remover: Optional[MeanDiffRemover]
    get_sensors: SensorProbe
    md_remover: Optional[HookingMeanDiffRemover]
    bdas: Optional[BoundlessDasLayer]

    def __init__(
        self,
        model,
        get_embeds_for_sensors: Optional[torch.nn.Module] = None,
        get_sensors: Optional[SensorProbe] = None,
        probe_dim: int = 3,
        get_embeds_mode: GetEmbedsMode = "identity",
        get_sensors_mode: GetSensorsMode = "linear",
        intermediate_layer_md_remover: Optional[HookingMeanDiffRemover] = None,
        bdas: Optional[BoundlessDasLayer] = None,
        general_layer_md_remover_args: Optional[GeneralLayerMdRemoverArgs] = None,
        sensor_md_remover: Optional[MeanDiffRemover] = None,
        sensor_remover_args: Optional[GeneralMdRemoverArgs] = None,
        bdas_after_layer_with_n_sensors: Optional[tuple[int, int]] = None,
        seq_len: int = 1023,
    ):
        super().__init__()
        self.model = model

        self.get_embeds_for_sensors = (
            get_embeds_for_sensors
            if get_embeds_for_sensors is not None
            else init_embeds_for_sensors(get_embeds_mode, get_hidden_size(model)).to(model.device)
        )
        if sensor_md_remover is not None:
            self.sensor_md_remover = sensor_md_remover
        elif sensor_remover_args is not None:
            self.sensor_md_remover = MeanDiffRemover(
                get_hidden_size(model), remover_args=sensor_remover_args, embedding_count=probe_dim
            )
        else:
            self.sensor_md_remover = None

        self.get_sensors = (
            get_sensors
            if get_sensors is not None
            else init_answer(get_sensors_mode, get_hidden_size(model), probe_dim).to(model.device)
        )

        # md_remover
        if intermediate_layer_md_remover is not None:
            self.md_remover = intermediate_layer_md_remover
        elif general_layer_md_remover_args is not None:
            remover_base = MeanDiffRemover(
                get_hidden_size(model),
                remover_args=general_layer_md_remover_args.remover_args,
                embedding_count=seq_len,
            )

            self.md_remover = HookingMeanDiffRemover(remover_base, general_layer_md_remover_args.layer)
        else:
            self.md_remover = None

        # bdas
        if bdas is not None:
            self.bdas = bdas
        elif bdas_after_layer_with_n_sensors is not None:
            self.bdas = BoundlessDasLayer(get_hidden_size(model), *bdas_after_layer_with_n_sensors, seq_len=seq_len)
        else:
            self.bdas = None

    def init_to_toks(self, yes_tok: int, no_tok: int):
        self.get_sensors.weight.data[:, :] = self.lm_head.weight.data[yes_tok] - self.lm_head.weight.data[no_tok]
        if hasattr(self.lm_head, "bias") and self.lm_head.bias is not None:
            self.get_sensors.bias.data[:] = self.lm_head.bias.data[yes_tok] - self.lm_head.bias.data[no_tok]

    def forward(self, input_ids, attention_mask, batch_data: BatchData):
        null_context = contextlib.nullcontext
        with self.bdas.hook_into(self.model, batch_data) if self.bdas is not None else null_context():
            with self.md_remover.hook_into(self.model, batch_data) if self.md_remover is not None else null_context():
                out_v = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                # print(f"{out_v=}")
                final_embeds = out_v.last_hidden_state

        if (
            len(list(self.get_sensors.parameters())) > 0
            and next(self.get_sensors.parameters()).device != final_embeds.device
        ) or (
            len(list(self.lm_head.parameters())) > 0 and next(self.lm_head.parameters()).device != final_embeds.device
        ):
            print(f"Moving heads to device {final_embeds.device}")
            device = final_embeds.device
            self.lm_head.to(device)
            self.get_sensors.to(device)
            self.get_embeds_for_sensors.to(device)

        assert ~final_embeds.isnan().any()

        logits = self.call_lm_head(final_embeds)

        embeds_for_sensors = self.get_embeds_for_sensors(final_embeds)
        # b,s,d
        embeds_for_sensors_idxed = embeds_for_sensors[
            torch.arange(embeds_for_sensors.shape[0])[:, None].to(embeds_for_sensors.device), get_all_locs(batch_data)
        ]
        if self.sensor_md_remover is not None:
            embeds_for_sensors_idxed = self.sensor_md_remover(embeds_for_sensors_idxed, batch_data)
        sensors = self.get_sensors(embeds_for_sensors_idxed).contiguous()

        return logits, sensors

    @property
    def transformer(self):
        return get_transformer(self.model)

    @property
    def lm_head(self):
        return get_lm_head(self.model)

    def call_lm_head(self, x):
        return find_call_lm_head(self.model, self.lm_head, x)

    @classmethod
    def from_folder(
        cls,
        folder: str,
        get_embeds_mode: Union[GetEmbedsMode, Literal["undefined"]] = "undefined",
        get_sensors_mode: Union[GetSensorsMode, Literal["undefined"]] = "undefined",
        general_layer_md_remover_args: Union[Optional[GeneralLayerMdRemoverArgs], Literal["undefined"]] = "undefined",
        sensor_remover_args: Union[Optional[GeneralMdRemoverArgs], Literal["undefined"]] = "undefined",
        bdas_after_layer: Union[Optional[int], Literal["undefined"]] = "undefined",
    ):
        """Load a model from a folder.

        Args:
            folder: The folder to load from.
            TODO: document"""

        # Avoid circular import
        from measurement_tampering.settings import get_setting
        from measurement_tampering.train_fsdp import get_default_config

        parent_folder = Path(folder).parent
        config = get_default_config()
        if os.path.exists(f"{parent_folder}/config.json"):
            upd = json.load(open(f"{parent_folder}/config.json", "r"))
            config.update(upd)
            if "dataset_kind" not in upd:
                print("WARN: dataset_kind not in config.json. Maybe unrelated config?")
                print(f"{upd=}")
        setting = get_setting(config["dataset_kind"])

        orig_model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True)

        T = TypeVar("T")

        def extract_load(
            input_val: Union[T, Literal["undefined"]],
            config_val: T,
            get_func: Callable[[T], Optional[torch.nn.Module]],
            state_dict_name: str,
            is_eq_for_load: Callable[[T, T], bool] = lambda x, y: x == y,
        ) -> Optional[torch.nn.Module]:
            # TODO: when will this not be in config!!
            actual_val = config_val if input_val == "undefined" else input_val

            if actual_val is not None:
                out = get_func(actual_val)
                if input_val == "undefined" or is_eq_for_load(config_val, input_val):
                    out.load_state_dict(torch.load(f"{folder}/{state_dict_name}"))
            else:
                out = None

            print(f"{state_dict_name=} {input_val=} {config_val=} {is_eq_for_load(config_val, input_val)=}")

            return out

        get_embeds_for_sensors = extract_load(
            get_embeds_mode,
            config["embeds_mode"],
            lambda x: init_embeds_for_sensors(x, get_hidden_size(orig_model)),
            "get_embeds_state_dict.pt",
        )
        assert isinstance(get_embeds_for_sensors, torch.nn.Module)
        if config["use_sensor_md_remover"]:
            config_md_remover_args = GeneralMdRemoverArgs(
                config["sensor_md_remover_remove_labels"], config["sensor_md_remover_is_locked"]
            )
        else:
            config_md_remover_args = None

        sensor_md_remover = extract_load(
            sensor_remover_args,
            config_md_remover_args,
            lambda x: None
            if x is None
            else MeanDiffRemover(
                get_hidden_size(orig_model), remover_args=x, embedding_count=setting.nb_overall_sensor_values
            ),
            "sensor_md_remover_state_dict.pt",
            is_eq_for_load=lambda x, y: x is not None and y is not None and x.use_labels == y.use_labels,
        )
        assert isinstance(sensor_md_remover, MeanDiffRemover) or sensor_md_remover is None
        get_sensors = extract_load(
            get_sensors_mode,
            config["sensors_mode"],
            lambda x: init_answer(x, get_hidden_size(orig_model), setting.nb_overall_sensor_values),
            "get_sensors_state_dict.pt",
        )
        assert isinstance(get_sensors, SensorProbe)

        config_remove_after_layer = config.get("remove_after_layer", None)
        config_remove_labels = config.get("remove_labels", "is_correct")
        config_lock_remover = config.get("lock_remover", False)
        config_general_remover_args = GeneralMdRemoverArgs(config_remove_labels, config_lock_remover)
        if general_layer_md_remover_args == "undefined":
            general_layer_md_remover_args_v = (
                None
                if config_remove_after_layer is None
                else GeneralLayerMdRemoverArgs(config_remove_after_layer, config_general_remover_args)
            )

        else:
            general_layer_md_remover_args_v = general_layer_md_remover_args
        if general_layer_md_remover_args_v is not None:
            md_remover = HookingMeanDiffRemover(
                MeanDiffRemover(
                    get_hidden_size(orig_model),
                    remover_args=general_layer_md_remover_args_v.remover_args,
                    embedding_count=setting.seq_len - 1,
                ),
                general_layer_md_remover_args_v.layer,
            )

            # load if layer and labels match (not if lock_remover is different)
            if general_layer_md_remover_args_v.remover_args == config_general_remover_args:
                md_remover.remover.load_state_dict(torch.load(f"{folder}/md_remover_state_dict.pt"))

        else:
            md_remover = None

        config_bdas_after_layer = config.get("bdas_after_layer", None)
        if bdas_after_layer == "undefined":
            bdas_after_layer = config_bdas_after_layer
        if bdas_after_layer is not None:
            bdas = BoundlessDasLayer(
                get_hidden_size(orig_model),
                bdas_after_layer,
                setting.nb_individual_sensor_values,
                seq_len=setting.seq_len - 1,
            )
            if config_bdas_after_layer == bdas_after_layer:
                bdas.load_state_dict(torch.load(f"{folder}/bdas_state_dict.pt"))
        else:
            bdas = None

        model = ModelWithAnswerPred(
            orig_model,
            get_embeds_for_sensors=get_embeds_for_sensors,
            sensor_md_remover=sensor_md_remover,
            get_sensors=get_sensors,
            intermediate_layer_md_remover=md_remover,
            bdas=bdas,
        )

        return model


def init_embeds_for_sensors(get_embeds_mode: GetEmbedsMode, n_embd: int):
    if get_embeds_mode == "identity":
        return torch.nn.Identity()
    elif get_embeds_mode == "small_attn":
        return WeightingAttn(n_embd, qk_dim=16)
    elif get_embeds_mode == "full_attn":
        return WeightingAttn(n_embd, qk_dim=n_embd)
    else:
        raise ValueError(f"Unknown {get_embeds_mode=}")


def init_answer(get_sensors_mode: GetSensorsMode, n_embd: int, probe_dim: int):
    if get_sensors_mode == "linear":
        return LinearProbe(n_embd, probe_dim)
    elif get_sensors_mode == "linear_shared":
        return LinearShared(n_embd, probe_dim)
    else:
        raise ValueError(f"Unknown {get_sensors_mode=}")


def project(t, d):
    """Project t onto the hyperplane orthogonal to d."""
    d = d / (torch.norm(d) + 1e-6)
    return t - torch.einsum("bld,d,h->blh", t, d, d)


def project_per_position(t, d):
    """Project t onto the hyperplane orthogonal to d."""
    d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-6)
    return t - torch.einsum("bld,ld,lh->blh", t, d, d)


def test_projection():
    r = torch.randn((3, 20, 10))
    d = torch.randn((10,))
    projection = project(r, d)
    dot_products = torch.einsum("bld,d->bl", projection, d)
    assert torch.allclose(dot_products, torch.zeros_like(dot_products), atol=1e-5)


def test_projection_per_position():
    r = torch.randn((3, 20, 10))
    d = torch.randn((20, 10))
    projection = project_per_position(r, d)
    dot_products = torch.einsum("bld,ld->bl", projection, d)
    assert torch.allclose(dot_products, torch.zeros_like(dot_products), atol=1e-5)


def test_md_remover():
    key = "a"
    h_dim = 10
    seq_len = 8
    mean0 = torch.randn((seq_len, h_dim))
    mean1 = torch.randn((seq_len, h_dim))

    def get_batch(batch_size):
        nb_positives = random.randint(0, batch_size)
        positives = torch.randn((nb_positives, seq_len, h_dim)) + mean0
        negatives = torch.randn((batch_size - nb_positives, seq_len, h_dim)) + mean1
        full = torch.cat([positives, negatives], dim=0)
        batch_data = {key: torch.cat([torch.zeros((nb_positives,)), torch.ones((batch_size - nb_positives,))]).bool()}
        shuffle_indexs = torch.randperm(batch_size)
        return (full[shuffle_indexs], {k: v[shuffle_indexs] for k, v in batch_data.items()})

    batches = [get_batch(400) for _ in tqdm(range(100))]
    md_remover = MeanDiffRemover(h_dim, 0, key, embedding_count=seq_len)
    for batch, batch_data in tqdm(batches):
        md_remover(batch, batch_data)
    torch.testing.assert_close(md_remover.mean0, mean0, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(md_remover.mean1, mean1, atol=5e-2, rtol=5e-2)

    # normalized_dir = (mean1 - mean0) / torch.norm(mean1 - mean0, dim=-1, keepdim=True)
    # bias = torch.einsum("ld,ld->l", (mean0 + mean1) / 2, normalized_dir)
    # torch.testing.assert_close(md_remover.bias, bias, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    test_projection()
    test_projection_per_position()
    test_md_remover()
