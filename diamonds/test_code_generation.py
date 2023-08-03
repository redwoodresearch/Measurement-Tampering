import multiprocessing
import random

from diamonds.code_generation import fill_template, sample_action_templates


def test_fill_template():
    assert fill_template('Hello "{2 + 3}" World', {}) == "Hello 5 World"
    assert fill_template('Numbers: "{1 + 1}",\n"{2 * 2}",\n"{3 ** 2}"', {}) == "Numbers: 2,\n4,\n9"
    assert fill_template("No match", {}) == "No match"
    assert fill_template('Nested {"{1 + 1}"}', {}) == "Nested {2}"
    assert fill_template('Multiple "{1 + 1}" and "{2 * 2}"', {}) == "Multiple 2 and 4"
    assert fill_template('Multiple "{random.randint(2, 2)}" and "{2 * 2}"', {}) == "Multiple 2 and 4"
    assert fill_template('a="{a}"', {"a": 5}) == "a=5"


def test_no_leak():
    kinds = ["easy", "hard", "both", "val", "only_val"]

    results = [
        (sample_action_templates(who, kind, n=10, split_seed=1), kind, who)
        for who in ["robber", "protector"]
        for kind in kinds
        for _ in range(1000)
    ]

    results_per_kind = {k: set() for k in kinds}

    for r, k, w in results:
        for template in r:
            results_per_kind[k].add((template, w))

    assert len(results_per_kind["easy"] & results_per_kind["hard"]) == 0  # easy and hard are disjoint
    assert len(results_per_kind["both"].difference(results_per_kind["val"])) == 0  # both in val
    assert len(results_per_kind["val"]) > len(results_per_kind["both"])  # val strictly bigger than both
    assert len(results_per_kind["hard"].difference(results_per_kind["both"])) == 0  # hard in both
    assert len(results_per_kind["hard"]) < len(results_per_kind["both"])  # hard strictly smaller than both
    assert len(results_per_kind["easy"].difference(results_per_kind["both"])) == 0  # easy in both
    assert len(results_per_kind["easy"]) < len(results_per_kind["both"])  # easy strictly smaller than both
    assert len(results_per_kind["only_val"].difference(results_per_kind["val"])) == 0  # only_val in val
    assert len(results_per_kind["only_val"] & results_per_kind["easy"]) == 0  # only_val and both are disjoint
    assert len(results_per_kind["only_val"] & results_per_kind["hard"]) == 0  # only_val and both are disjoint
    assert len(results_per_kind["only_val"] & results_per_kind["both"]) == 0  # only_val and both are disjoint


if __name__ == "__main__":
    test_fill_template()
    test_no_leak()
