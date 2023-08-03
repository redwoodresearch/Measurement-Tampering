from __future__ import annotations

import torch
from attrs import define

from measurement_tampering.train_utils import TokenizedDataBaseSetting
from text_properties.simplified import sensors_simple


@define
class TextPropertiesSetting(TokenizedDataBaseSetting):
    nb_individual_sensor_values: int = len(sensors_simple)
    val_splits: list[str] = ["val"]
    answers_train_split: str = "train"
    seq_len: int = 1024

    def initialization_tokens(self, tokenizer) -> tuple[int, int]:
        """Return the token ids for the sensor initialization tokens"""
        return self.single_tokenize(" Yes", tokenizer), self.single_tokenize(" No", tokenizer)

    def highlighted_ntp_metrics(self, next_token_mask: torch.Tensor, token_loss_by_token: torch.Tensor):
        sensor_locs = next_token_mask == 100.0  # HACK
        if sensor_locs.any():
            average_sensor_hack_loss = token_loss_by_token[sensor_locs].sum() / sensor_locs.sum()
        else:
            average_sensor_hack_loss = torch.tensor(0.0).to(sensor_locs)

        return {
            "sensor_hack_l": average_sensor_hack_loss.float(),
        }
