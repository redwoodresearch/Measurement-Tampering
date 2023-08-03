from __future__ import annotations

from attrs import define

from measurement_tampering.train_utils import TokenizedDataBaseSetting
from money_making_easy.basic_setup import sensor_text_list


@define
class MoneyMakingNewSetting(TokenizedDataBaseSetting):
    nb_individual_sensor_values: int = len(sensor_text_list)
    val_splits: list[str] = ["val"]
    answers_train_split: str = "train"
    seq_len: int = 512 * 3

    def initialization_tokens(self, tokenizer) -> tuple[int, int]:
        """Return the token ids for the sensor initialization tokens"""
        return self.single_tokenize(" Yes", tokenizer), self.single_tokenize(" No", tokenizer)
