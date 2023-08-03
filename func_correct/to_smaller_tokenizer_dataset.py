# %%
import os
from math import ceil, floor

import torch

torch.manual_seed(0)

base_dir = os.path.expanduser("~/code_elk_setting/correctness_data/full_p_extended_notrunc")
input_data_dir = f"{base_dir}/tokenized/for_codegen_tokenizer"
output_data_dir = f"{base_dir}/smaller_tokenized/for_codegen_tokenizer"
os.makedirs(output_data_dir, exist_ok=True)
file_names = ["train.pt", "non_overlapping_val.pt"]
files = [torch.load(f"{input_data_dir}/{file_name}") for file_name in file_names]
# %%
file = files[0]
print(*file.keys())
print(file["input_ids"].shape)
dirty_positive_mask = file["all_passes"] & ~file["is_clean"]
nano_mask = dirty_positive_mask & file["is_correct"]
tamper_mask = dirty_positive_mask & ~file["is_correct"]
print(len(nano_mask), tamper_mask.sum(), nano_mask.sum())


# %%
def take(data, mask):
    return {key: value[mask] for key, value in data.items()}


def take_tamper(data):
    return take(data, data["all_passes"] & ~data["is_correct"])


def take_true_pos(data):
    return take(data, data["all_passes"] & data["is_correct"])


def take_full_neg(data):
    return take(data, ~(data["passes"] & data["sensor_mask"]).any(-1))


def take_half_neg(data):
    return take(data, (data["passes"] & data["sensor_mask"]).any(-1) & ~data["all_passes"])


def take_kind(data, kind, clean):
    data = take(data, data["is_clean"] == clean)
    if kind == "tamper":
        return take_tamper(data)
    elif kind == "true_pos":
        return take_true_pos(data)
    elif kind == "full_neg":
        return take_full_neg(data)
    elif kind == "half_neg":
        return take_half_neg(data)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def balance(data, clean_args, dirty_args, prop_clean=0.1):
    clean_args["half_neg"] = 1 - sum(clean_args.values())
    dirty_args["half_neg"] = 1 - sum(dirty_args.values())
    n = len(data["input_ids"])
    sorted_datas = {
        (clean, kind): take_kind(data, kind, clean)
        for clean in [True, False]
        for kind in ["tamper", "true_pos", "full_neg", "half_neg"]
    }
    for k, v in sorted_datas.items():
        print(k, len(v["input_ids"]), "=", len(v["input_ids"]) / n)
    assert n == sum(len(v["input_ids"]) for v in sorted_datas.values())
    undersampling_factors = {
        (clean, kind): len(sorted_datas[(clean, kind)]["input_ids"]) / ceil(args[kind] * n * p)
        for clean, args, p in [(True, clean_args, prop_clean), (False, dirty_args, 1 - prop_clean)]
        for kind in ["tamper", "true_pos", "full_neg", "half_neg"]
        if args.get(kind, 0) > 0
    }
    for k, v in undersampling_factors.items():
        print(k, v)
    undersampling_factor = min(undersampling_factors.values())
    print(f"Undersampling factor: {undersampling_factor}")
    permutations = {k: torch.randperm(len(v["input_ids"])) for k, v in sorted_datas.items()}
    balanced_datas = {
        field: torch.cat(
            [
                sorted_datas[(clean, kind)][field][permutations[(clean, kind)]][
                    : floor(undersampling_factor * args[kind] * n * p)
                ]
                for clean, args, p in [(True, clean_args, prop_clean), (False, dirty_args, 1 - prop_clean)]
                for kind in ["tamper", "true_pos", "full_neg", "half_neg"]
                if args.get(kind, 0) > 0
            ]
        )
        for field in data.keys()
    }
    new_n = len(balanced_datas["input_ids"])
    print(f"Kept {new_n} / {n} = {new_n / n:.2%} of the data")
    final_perm = torch.randperm(new_n)
    shuffled_balanced_data = {key: value[final_perm] for key, value in balanced_datas.items()}
    new_sorted_datas = {
        (clean, kind): take_kind(shuffled_balanced_data, kind, clean)
        for clean in [True, False]
        for kind in ["tamper", "true_pos", "full_neg", "half_neg"]
    }
    assert new_n == sum(len(v["input_ids"]) for v in new_sorted_datas.values())
    new_undersampling_factors = {
        (clean, kind): len(new_sorted_datas[(clean, kind)]["input_ids"]) / ceil(args[kind] * new_n * p)
        for clean, args, p in [(True, clean_args, prop_clean), (False, dirty_args, 1 - prop_clean)]
        for kind in ["tamper", "true_pos", "full_neg", "half_neg"]
        if args.get(kind, 0) > 0
    }
    new_undersampling_factor = min(new_undersampling_factors.values())
    print(f"New undersampling factor: {new_undersampling_factor}")
    return shuffled_balanced_data


clean_args = {
    "true_pos": 0.8,
    "full_neg": 0.2,
}
dirty_args = {
    "tamper": 0.05,
    "true_pos": 0.4,
    "full_neg": 0.25,
}
clean_val_args = {
    "true_pos": 0.8,
    "full_neg": 0.2,
}
dirty_val_args = {
    "tamper": 0.4,
    "true_pos": 0.4,
    "full_neg": 0.1,
}
train, val = files
new_train = balance(train, clean_args, dirty_args)
new_val = balance(val, clean_val_args, dirty_val_args)
# %%
torch.save(new_train, f"{output_data_dir}/train.pt")
torch.save(new_val, f"{output_data_dir}/non_overlapping_val.pt")

# %%
