import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from diamonds.train_data_gen import OMIT_TOKEN
from measurement_tampering.train_utils import BatchData

parent_folder = "~/rrfs/elk/diamonds/v3.7/data"

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
pad_token = " ."
max_length = 1024

tokenizer.padding_side = "left"
tokenizer.pad_token = pad_token
pad_token_ids = tokenizer.encode(pad_token)
assert len(pad_token_ids) == 1
tokenizer.pad_token_id = pad_token_ids[0]

split_to_file = {
    "train": "answers_train.pt",
    "validation": "answers_val.pt",
}
for seed in range(8):
    folder = os.path.expanduser(f"{parent_folder}/s{seed}")
    dataset = load_dataset(f"redwoodresearch/diamonds-seed{seed}")
    for split, ds in dataset.items():
        texts = ["# SENSOR:\n" + text for text in ds["text"]]
        input_ids = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids

        untruncated = input_ids[:, 0] == tokenizer.pad_token_id

        print(f"{split}: {(untruncated == False).sum()} / {len(untruncated)} skipped because truncated")

        input_ids = input_ids[untruncated]
        is_correct = torch.tensor(ds["is_correct"])[untruncated]
        is_clean = torch.tensor(ds["is_clean"])[untruncated]
        sensor_values = torch.tensor(ds["measurements"])[untruncated]
        difficulty = torch.tensor(ds["difficulty"])[untruncated]

        batch, seq_len = input_ids.shape
        _, nb_sensors = sensor_values.shape

        omit_mask = input_ids == tokenizer.encode(OMIT_TOKEN)[0]
        assert (omit_mask.sum(1) == nb_sensors).all(), f"{omit_mask.sum(1)} != {nb_sensors}"
        sensor_pos = omit_mask.nonzero(as_tuple=True)[1].reshape(-1, nb_sensors)
        assert (sensor_pos[1:, :] == sensor_pos[:-1, :]).all(), "All sensors should be in the same position"

        torch.save(
            BatchData(
                passes=sensor_values,
                all_passes=sensor_values.all(dim=1),
                is_correct=is_correct,
                is_clean=is_clean,
                input_ids=input_ids,
                attention_mask=torch.ones((batch, seq_len), dtype=torch.bool),
                sensor_locs=sensor_pos,
                sensor_mask=torch.ones((batch, nb_sensors), dtype=torch.bool),
                overall_loc=torch.full((batch,), seq_len - 2, dtype=torch.long),
                ntp_mask=torch.zeros((batch, seq_len), dtype=torch.bool),
                difficulty=difficulty,
            ),
            f"{folder}/{split_to_file[split]}",
        )
