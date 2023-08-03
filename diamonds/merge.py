import os

import torch

from measurement_tampering.train_utils import BatchData


def run(file_name_1: str, file_name_2: str, output_file: str, seed: int = 0):
    file_name_1 = os.path.expanduser(file_name_1)
    file_name_2 = os.path.expanduser(file_name_2)
    output_file = os.path.expanduser(output_file)

    torch.manual_seed(seed)

    data1: BatchData = torch.load(file_name_1)
    data2: BatchData = torch.load(file_name_2)

    keys_1 = set(data1.keys())
    keys_2 = set(data2.keys())
    assert keys_1 == keys_2, f"keys_1: {keys_1}, keys_2: {keys_2}"
    for k in keys_1:
        assert data1[k].shape[1:] == data2[k].shape[1:], f"Shape mismatch for {k}: {data1[k].shape} vs {data2[k].shape}"
    for k in keys_1:
        assert len(data1[k]) == len(
            data1["input_ids"]
        ), f"Length mismatch for {k}: {len(data1[k])} vs {len(data1['input_ids'])}"
    for k in keys_2:
        assert len(data2[k]) == len(
            data2["input_ids"]
        ), f"Length mismatch for {k}: {len(data2[k])} vs {len(data2['input_ids'])}"
    print("Merging data with keys:", keys_1)

    perm = torch.randperm(len(data2["input_ids"]) + len(data1["input_ids"]))
    torch.save({k: torch.cat([v, data2[k]], dim=0)[perm] for k, v in data1.items()}, output_file)


if __name__ == "__main__":
    from fire import Fire

    Fire(run)
