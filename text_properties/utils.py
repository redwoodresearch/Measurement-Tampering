from typing import Optional

import torch

from measurement_tampering.train_utils import BatchData
from text_properties.simplified import sensors_simple


def batch_data_from_ntp(
    input_ids: torch.Tensor,
    pad_tok: int,
    cut_off_tokens_mask: Optional[torch.Tensor] = None,
    mask_out_prompting_weight: Optional[float] = None,
    probe_loc_tokens: Optional[torch.Tensor] = None,
    sensor_values: Optional[torch.Tensor] = None,
    actual_latent: Optional[torch.Tensor] = None,
    is_clean: Optional[torch.Tensor] = None,
):
    batch, seq_len = input_ids.shape

    if cut_off_tokens_mask is not None:
        print(f"{input_ids.shape} {cut_off_tokens_mask.shape=}")
        # on *after* cut off tokens
        matches = (
            torch.stack(
                [
                    input_ids[:, i : seq_len - (cut_off_tokens_mask.shape[0] - i)]
                    for i in range(cut_off_tokens_mask.shape[0])
                ],
                dim=-1,
            )
            == cut_off_tokens_mask[None, None, :]
        ).all(dim=-1)
        # assert ((matches.sum(dim=-1, dtype=torch.long) == 0) | (matches.sum(dim=-1, dtype=torch.long) == 1)).all()

        ntp_mask = torch.cat(
            [
                torch.zeros((batch, cut_off_tokens_mask.shape[0]), dtype=torch.bool),
                torch.cummax(matches, dim=-1).values,
            ],
            dim=-1,
        )
        if mask_out_prompting_weight is not None:
            ntp_mask = torch.where(ntp_mask, torch.tensor(1.0), torch.tensor(mask_out_prompting_weight))
    else:
        ntp_mask = torch.full((batch, seq_len), True, dtype=torch.bool)

    if probe_loc_tokens is not None:
        probe_loc_matches = (
            torch.stack(
                [input_ids[:, i : seq_len - (probe_loc_tokens.shape[0] - i)] for i in range(probe_loc_tokens.shape[0])],
                dim=-1,
            )
            == probe_loc_tokens[None, None, :]
        ).all(dim=-1)
        assert probe_loc_matches.any(dim=-1).all()
        locs = torch.argmax(probe_loc_matches.to(torch.uint8), dim=-1) + probe_loc_tokens.shape[-1] - 1
        locs = torch.where(probe_loc_matches.any(dim=-1), locs, input_ids.shape[-1] - 3)
    else:
        is_pad = input_ids == pad_tok
        locs = torch.where(
            is_pad.any(dim=-1), torch.argmax(is_pad.to(torch.uint8), dim=-1) - 2, input_ids.shape[-1] - 3
        )
    # same loc for all sensors
    sensor_locs = torch.stack([locs] * len(sensors_simple), dim=-1)
    overall_loc = locs

    return BatchData(
        passes=sensor_values if sensor_values is not None else torch.zeros((batch, 3), dtype=torch.bool),
        all_passes=sensor_values.all(dim=-1) if sensor_values is not None else torch.zeros(batch, dtype=torch.bool),
        is_correct=actual_latent if actual_latent is not None else torch.zeros(batch, dtype=torch.bool),
        is_clean=is_clean if is_clean is not None else torch.full((batch,), True, dtype=torch.bool),
        input_ids=input_ids,
        attention_mask=torch.ones((batch, seq_len), dtype=torch.bool),
        sensor_locs=sensor_locs,
        sensor_mask=torch.full(sensor_locs.shape, True),
        overall_loc=overall_loc,
        ntp_mask=ntp_mask,
        difficulty=torch.full((batch,), 0, dtype=torch.long),
        counterfactual_tokens=torch.zeros((batch, 0), dtype=torch.long),
        swaps=torch.zeros((batch, 0), dtype=torch.bool),
    )


def batch_data_from_strings_tokenizer(
    strs: list[str],
    tokenizer,
    mask_out_prompting: bool = False,
    max_len: int = 512 * 3,
    mask_out_prompting_weight: Optional[float] = None,
    sensor_values: Optional[list[dict[str, bool]]] = None,
    actual_latent: Optional[list[bool]] = None,
    is_clean: Optional[list[bool]] = None,
    prob_at_end_instead_of_modify: bool = False,
):
    out = tokenizer(strs, padding="max_length", max_length=max_len, return_tensors="pt", truncation=True)

    modified_text_point = tokenizer("Modified text:\n```", return_tensors="pt")["input_ids"].squeeze(0)
    assert len(tokenizer.encode("[PAD]")) == 1

    return batch_data_from_ntp(
        out["input_ids"],
        tokenizer.encode("[PAD]")[0],
        modified_text_point if mask_out_prompting else None,
        mask_out_prompting_weight=mask_out_prompting_weight,
        probe_loc_tokens=None if prob_at_end_instead_of_modify else modified_text_point,
        sensor_values=torch.tensor([[d[s] for s in sensors_simple] for d in sensor_values])
        if sensor_values is not None
        else None,
        actual_latent=torch.tensor(actual_latent) if actual_latent is not None else None,
        is_clean=torch.tensor(is_clean) if is_clean is not None else None,
    )
