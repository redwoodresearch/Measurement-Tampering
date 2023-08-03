# from rust_circuit.ui.very_named_tensor import VeryNamedTensor
from typing import Optional

from attrs import define

from diamonds.code_generation import NB_SENSORS
from measurement_tampering.train_utils import TokenizedDataBaseSetting, TokenizedDataElkDataset


@define
class DiamondsSetting(TokenizedDataBaseSetting):
    nb_individual_sensor_values: int = NB_SENSORS
    val_splits: list[str] = ["answers_val", "answers_val_train"]
    answers_train_split: str = "answers_train"
    seq_len: int = 1024

    def _single_tokenize(self, x: str, tokenizer):
        out = tokenizer(x)["input_ids"]
        assert len(out) == 1
        return out[0]

    def initialization_tokens(self, tokenizer) -> tuple[int, int]:
        """Return the token ids for the sensor initialization tokens"""
        return self._single_tokenize("5", tokenizer), self._single_tokenize("0", tokenizer)
