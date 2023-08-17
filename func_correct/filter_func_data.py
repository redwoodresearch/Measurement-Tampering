# %%
import os.path
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from func_correct.apply_model_to_data import FuncCorrectSetting
from func_correct.loaded_problem import get_converter

# %%
setting = FuncCorrectSetting()


def do_filter(data_dir, out_data_dir):
    out_path = Path(out_data_dir).expanduser()
    try:
        out_path.mkdir()
    except FileExistsError:
        raise ValueError(f"Output directory {out_path} already exists")

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    data_dir_here = os.path.expanduser(data_dir)

    for data_split in ["non_overlapping_val", "train"]:
        data = setting.load_data(data_dir_here, data_split)
        print(f"Loaded {len(data)} examples from {data_dir}/{data_split}")
        good = [item["input_ids"][-1] == tokenizer.pad_token_id for item in tqdm(data)]
        to_save = [item for item, g in zip(data.data, good) if g]
        print(f"Filtered to {len(to_save)} examples")

        json_converter = get_converter()
        out_file = out_path / f"{data_split}_data.jsonl"
        with open(out_file, "w") as f:
            for item in tqdm(to_save):
                f.write(json_converter.dumps(item) + "\n")


# %%

do_filter(
    data_dir="~/code_elk_setting/correctness_data/full_p_extended",
    out_data_dir="~/code_elk_setting/correctness_data/full_p_extended_notrunc",
)
