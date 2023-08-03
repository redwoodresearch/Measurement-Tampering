import os

import torch


def get_row_hash_set(m: torch.Tensor) -> set[float]:
    assert m.ndim == 2
    hashes = [hash(tuple(row.tolist())) for row in m]
    hash_set = set(hashes)
    assert len(hash_set) == len(hashes), f"Hash fn too weak... {len(hash_set)} vs {len(hashes)}"
    return hash_set


def check_subset(a: set, b: set):
    assert len(a.intersection(b)) == len(a)


def check_disjoint(a: set, b: set):
    assert len(a.intersection(b)) == 0


def load(folder, name):
    return get_row_hash_set(torch.load(os.path.join(folder, name))["input_ids"])


def check_leakages(folder):
    train = load(folder, "answers_train.pt")

    val = load(folder, "answers_val.pt")
    val_train = load(folder, "answers_val_train.pt")

    check_disjoint(train, val)
    check_subset(val_train, train)

    val_both = load(folder, "answers_val_both.pt")
    val_val = load(folder, "answers_val_val.pt")

    check_disjoint(val_val, val_both)
    check_subset(val_both, val)
    check_subset(val_val, val)


if __name__ == "__main__":
    from fire import Fire

    Fire(check_leakages)
    # check_leakages(os.path.expanduser("~/rrfs/elk/diamonds/v3.7/data/s1"))
