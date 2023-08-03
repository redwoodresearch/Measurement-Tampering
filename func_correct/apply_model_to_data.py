from attrs import define

from measurement_tampering.train_utils import TokenizedDataBaseSetting

MAX_ANSWER_COUNTS = 20


@define
class FuncCorrectSetting(TokenizedDataBaseSetting):
    nb_individual_sensor_values: int = MAX_ANSWER_COUNTS
    val_splits: list[str] = ["non_overlapping_val"]
    answers_train_split: str = "train"
    seq_len: int = 2048

    def initialization_tokens(self, tokenizer) -> tuple[int, int]:
        """Return the token ids for the sensor initialization tokens"""

        return self.single_tokenize(" pass", tokenizer), self.single_tokenize(" fail", tokenizer)
