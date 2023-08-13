# %%
import os.path
from pathlib import Path
from typing import Literal

import torch
from tqdm.auto import tqdm

from func_correct.apply_model_to_data import FuncCorrectSetting
from measurement_tampering.eval_models import epoch_str, get_scores, load_tokenizer_and_config
from func_correct.final_datum import FinalDatum
from func_correct.loaded_problem import get_converter

# %%
model_dir = "~/unity_dup/big_model_new_lr"
data_dir = "~/rrfs/code_elk_setting/correctness_data/full_p_extended_notrunc"
out_data_dir = "~/rrfs/code_elk_setting/correctness_data/full_p_extended_doable"
filter_op = "doable"

# %%

setting = FuncCorrectSetting()


def do_filter(model_dir, data_dir, out_data_dir, filter_op: Literal["notrunc", "doable"]):
    out_path = Path(out_data_dir).expanduser()
    try:
        out_path.mkdir()
    except FileExistsError:
        raise ValueError(f"Output directory {out_path} already exists")

    model_folder = os.path.expanduser(model_dir)
    epoch = 1
    tokenizer, config = load_tokenizer_and_config(model_folder, epoch_str(epoch))
    seq_len = config["seq_len"] or 1024

    setting.create_get_losses(
        tokenizer,
        seq_len=seq_len,
        pad_left_with_dots=config["pad_left_with_dots"],
    )  # initialize the setting

    data_dir_here = os.path.expanduser(data_dir)

    for data_split in ["non_overlapping_val", "overlapping_val", "train"]:
        data = setting.load_data(data_dir_here, data_split)
        print(f"Loaded {len(data)} examples from {data_dir}/{data_split}")
        if filter_op == "notrunc":
            # Filter out truncated examples:
            # just check whether they end with a pad token
            # (rules out a few extra examples, but that's fine)
            good = [item["input_ids"][-1] == tokenizer.pad_token_id for item in tqdm(data)]
        elif filter_op == "doable":
            # Get model scores
            scores = get_scores(data_dir_here, model_folder, epoch, batch_size=8, tmp=False, split=data_split)
            # make sure things match up
            data_answer_mask = torch.stack([item["padded_answer_locs"] >= 0 for item in data])
            assert torch.equal(data_answer_mask, scores["answer_mask"])
            # Filter out examples that the model gets wrong
            logit_threshold = 0.5  # go a bit up with the threshold so we don't lose too much tamper
            model_pred = scores["answer_correct"] > logit_threshold
            good = model_pred == scores["is_correct"]
        to_save = [item for item, g in zip(data.data, good) if g]
        print(f"Filtered to {len(to_save)} examples")

        json_converter = get_converter()
        out_file = out_path / f"{data_split}_data.json"
        with open(out_file, "w") as f:
            for item in tqdm(to_save):
                f.write(json_converter.dumps(item) + "\n")


# %%

do_filter(
    model_dir="~/unity_dup/big_model_new_lr",
    data_dir="~/code_elk_setting/correctness_data/full_p_extended",
    out_data_dir="~/rrfs/code_elk_setting/correctness_data/full_p_extended_notrunc",
    filter_op="notrunc",
)

# %%

do_filter(
    model_dir="~/unity_dup/big_model_new_lr",
    data_dir="~/rrfs/code_elk_setting/correctness_data/full_p_extended_notrunc",
    out_data_dir="~/rrfs/code_elk_setting/correctness_data/full_p_extended_doable",
    filter_op="doable",
)
