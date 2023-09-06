import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from measurement_tampering.train_utils import BatchData
from text_properties.training_data_utils import get_sensor_locs

# %%

is_easy = False

ds_name = "text_properties"
dataset = load_dataset(f"redwoodresearch/{ds_name}")

# %%

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# %%

max_length = 1024

# %%

output_dir = "./"

for split, ds in dataset.items():
    texts = ds["text"]
    input_ids = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
        truncation=True,
    ).input_ids

    untruncated = input_ids[:, -1] == tokenizer.pad_token_id

    print(
        f"{split}: {(untruncated == False).sum()} / {len(untruncated)} skipped because truncated"
    )

    input_ids = input_ids[untruncated]
    is_correct = torch.tensor(ds["is_correct"])[untruncated]
    is_clean = torch.tensor(ds["is_clean"])[untruncated]
    sensor_values = torch.tensor(ds["measurements"])[untruncated]

    batch, seq_len = input_ids.shape
    _, nb_sensors = sensor_values.shape

    sensor_locs = get_sensor_locs(input_ids, tokenizer)

    torch.save(
        BatchData(
            passes=sensor_values,
            all_passes=sensor_values.all(dim=1),
            is_correct=is_correct,
            is_clean=is_clean,
            input_ids=input_ids,
            attention_mask=torch.ones((batch, seq_len), dtype=torch.bool),
            sensor_locs=sensor_locs,
            sensor_mask=torch.ones((batch, nb_sensors), dtype=torch.bool),
            overall_loc=sensor_locs[:, -1],
            ntp_mask=torch.full((batch, seq_len), True, dtype=torch.bool),
            difficulty=torch.full((batch,), 0, dtype=torch.long),
            counterfactual_tokens=torch.zeros((batch, 0), dtype=torch.long),
            swaps=torch.zeros((batch, 0), dtype=torch.bool),
        ),
        f"{output_dir}/{split}.pt",
    )

# %%

# this is code which tests the equivalence to an existing version of the file.

# x = torch.load(
#     os.path.expanduser(
#         "~/rrfs/ryan/text_properties/simplified_setting_v3/post_pretrain_with_extra_tamp_sensor_pred/data_gpt_neo_x_tokenizer/train.pt"
#     )
# )
# y = torch.load(f"{output_dir}/train.pt")

# assert x.keys() == y.keys()
# for k in x:
#     if k == "input_ids":
#         texts_old = [tokenizer.decode(s) for s in x[k]]

#         input_ids_old = tokenizer(
#             texts_old,
#             max_length=max_length,
#             padding="max_length",
#             return_tensors="pt",
#             truncation=True,
#         ).input_ids
#         assert (input_ids_old == y[k]).all()
#     elif k != "sensor_locs" and k != "overall_loc":
#         assert (x[k] == y[k]).all(), k
