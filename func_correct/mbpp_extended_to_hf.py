# %%
import os

from tqdm import tqdm

from func_correct.loaded_problem import LoadedProblem, get_converter

problems: list[LoadedProblem] = []

with open(os.path.expanduser("~/rrfs/code_elk_setting/extended.bak_v1/raw_functions_v3.jsonl"), "r") as f:
    for line in tqdm(f):
        problems.append(get_converter().loads(line, LoadedProblem))

# %%
from datasets import Dataset

ds = {
    "description": [p.description for p in problems],
    "gpt4_solution": [p.solutions[0] for p in problems],
    "function_name": [p.test_cases.fn_name for p in problems],
    "test_cases": [p.test_cases.get_assertions() for p in problems],
}
dataset = Dataset.from_dict(ds)
dataset.push_to_hub("redwoodresearch/mbpp_extended", token=True)
# %%
