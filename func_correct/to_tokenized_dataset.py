import os
from copy import deepcopy

import torch
from transformers import AutoTokenizer

from func_correct.apply_model_to_data import MAX_ANSWER_COUNTS, FuncCorrectSetting
from func_correct.final_datum import PRE_ANSWER_STR, FinalDatum
from func_correct.loaded_problem import get_converter
from measurement_tampering.train_utils import BatchData

# %%


# thx gpt4
def pad_nonzero_indices(tensor: torch.Tensor, max_count: int, offset: int = 0, pad_value: int = -1):
    # Get the indices of non-zero elements
    indices = torch.nonzero(tensor, as_tuple=True)

    # Count the number of non-zero elements per row
    counts = torch.sum(tensor, dim=1)
    assert int(counts.max()) <= max_count

    # Create a mask for valid indices
    mask = torch.arange(max_count, device=tensor.device).expand(tensor.shape[0], max_count) < counts.unsqueeze(1)

    # Initialize the padded tensor with -1 (or any other padding value)
    padded_indices = torch.full((tensor.shape[0], max_count), pad_value, dtype=torch.long, device=tensor.device)

    # Fill the padded tensor with the non-zero indices using the mask
    padded_indices[mask] = indices[1] + offset

    return padded_indices


def tokenize_dataset(
    data: list[FinalDatum], input_tokenizer, pad_left_with_dots: bool = True, max_answer_count: int = MAX_ANSWER_COUNTS
) -> BatchData:
    setting = FuncCorrectSetting()
    tokenizer = deepcopy(input_tokenizer)

    pre_answer_toks = tokenizer(PRE_ANSWER_STR)["input_ids"]

    all_text = [d.text + tokenizer.eos_token for d in data]

    if pad_left_with_dots:
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.encode(".")[0]
        tokenizer.pad_token = "."
    else:
        all_text = [tokenizer.bos_token + t for t in all_text]
    out = dict(
        tokenizer(
            all_text,
            padding="max_length",
            max_length=setting.seq_len,
            truncation=True,
            return_tensors="pt",
        )
    )
    if pad_left_with_dots:
        # don't mask out to avoid being + allow attending to counterfactuals
        out["attention_mask"][:, :] = 1

        # exclude truncated
        is_not_truncated = out["input_ids"][:, -1] == tokenizer.eos_token_id
        data = [d for d, is_not_truncated_i in zip(data, is_not_truncated) if is_not_truncated_i]
        out = {k: v[is_not_truncated] for k, v in out.items()}

        print("Truncated", (~is_not_truncated).sum().item(), "examples out of", len(is_not_truncated))

    answer_at_loc = torch.stack(
        [out["input_ids"][:, i : (-len(pre_answer_toks)) + i] == tok for i, tok in enumerate(pre_answer_toks)],
        dim=0,
    ).all(dim=0)
    raw_sensor_locs = pad_nonzero_indices(
        answer_at_loc, offset=len(pre_answer_toks) - 1, max_count=max_answer_count, pad_value=-1
    )
    out["overall_loc"] = torch.max(raw_sensor_locs, dim=-1).values
    out["sensor_mask"] = raw_sensor_locs != -1
    out["sensor_locs"] = torch.where(out["sensor_mask"], raw_sensor_locs, 0)

    def get_passes(x: list[bool]):
        t = torch.tensor(x)
        return torch.cat([t, torch.full((MAX_ANSWER_COUNTS - t.shape[0],), False)], dim=0)

    out["passes"] = torch.stack([get_passes(d.passes) for d in data], dim=0)
    out["all_passes"] = torch.tensor([all(d.passes) for d in data])
    out["is_correct"] = torch.tensor([d.is_correct for d in data])
    out["is_clean"] = torch.tensor([d.is_clean() for d in data])
    out["ntp_mask"] = torch.full(out["input_ids"].shape, 1.0, dtype=torch.float)
    if pad_left_with_dots:
        pad_mask = (
            torch.arange(out["input_ids"].shape[1])[None, :]
            >= torch.argmin((out["input_ids"] == tokenizer.encode(".")[0]).to(torch.uint8), dim=-1)[:, None]
        )
        out["ntp_mask"] = pad_mask.float()
        # TODO: fix this dumb padding hack! We can't use " #" above so we do this here...
        toks = tokenizer.encode(" #")
        assert len(toks) == 1
        out["input_ids"] = torch.where(pad_mask, out["input_ids"], toks[0])

    for datum, masked, _ in zip(data, out["sensor_mask"], out["input_ids"]):
        # sad for loop
        assert len(datum.passes) >= masked.sum(), (len(datum.passes), masked.sum())

    return BatchData(out)


# %%


gpt_neox_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
codegen_tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

tokenizers = dict(gpt_neox_tokenizer=gpt_neox_tokenizer, codegen_tokenizer=codegen_tokenizer)

# %%

file_names = {
    "train": "train_data.json",
    "non_overlapping_val": "non_overlapping_val_data.json",
    "overlapping_val": "overlapping_val_data.json",
}

# %%

input_data_dir = os.path.expanduser("~/code_elk_setting/correctness_data/full_p_extended_notrunc/")
output_data_dir = f"{input_data_dir}/tokenized"

os.makedirs(output_data_dir, exist_ok=True)


json_converter = get_converter()

for split, file_name in file_names.items():
    with open(f"{input_data_dir}/{file_name}", "r") as f:
        data = [json_converter.loads(l, FinalDatum) for l in f.readlines()]
    for tok_name, tokenizer in tokenizers.items():
        full_out_dir = f"{output_data_dir}/for_{tok_name}/"
        os.makedirs(full_out_dir, exist_ok=True)
        file = f"{full_out_dir}/{split}.pt"
        print(f"{file=}")
        tokenized_data = tokenize_dataset(data, tokenizer)
        torch.save(tokenized_data, file)


# return FuncCorrectDataset(*self._tokenize_dataset(data), max_answer_count=MAX_ANSWER_COUNTS)

# %%

# full_out_dir = f"{output_data_dir}/for_gpt_neox_tokenizer/"
# file = f"{full_out_dir}/train.pt"
# out = torch.load(file)

# {k: v.shape for k, v in out.items()}

# model_new = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m-deduped")

# %%

# print(gpt_neox_tokenizer.decode(out["input_ids"][0]))
# gpt_neox_tokenizer.encode(" .")[0]

# %%

# x = gpt_neox_tokenizer.decode(tokenize_dataset(data[:2], gpt_neox_tokenizer)["input_ids"][0])
# print(x)

# %%

# x_out = model_new(input_ids=out["input_ids"][:4], attention_mask=out["attention_mask"][:4])

# x_out["logits"].isnan().any()
# tok_new = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")
# tok_new.padding_side = "left"
# tok_new.pad_token_id = tok_new.encode(".")[0]
# tok_new.pad_token = "."


# out = tok_new(
#     "hi",
#     padding="max_length",
#     max_length=17,
#     truncation=True,
#     return_tensors="pt",
# )
# out
