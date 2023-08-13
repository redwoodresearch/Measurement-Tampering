import os

from datasets import load_dataset

from func_correct.final_datum import FinalDatum
from func_correct.loaded_problem import DATA_DIR, get_converter

save_jsonl_path = os.path.expanduser(f"{DATA_DIR}/correctness_data/full_p_extended_notrunc")
os.makedirs(save_jsonl_path, exist_ok=True)
ds = load_dataset("redwoodresearch/function_correctness")

json_converter = get_converter()

for split_name, file_name in {"train": "train_data.json", "validation": "non_overlapping_val_data.json"}.items():
    with open(f"{save_jsonl_path}/{file_name}", "w") as f:
        for d in ds[split_name]:
            obj = FinalDatum(d["text"], d["measurements"], d["is_correct"], d["is_clean"])
            f.write(json_converter.dumps(obj) + "\n")

import func_correct.to_tokenized_dataset # notebook which tokenizes the data
import func_correct.to_smaller_tokenizer_dataset # notebook which upsamples the data