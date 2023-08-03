# %%
import json
import os
from collections import Counter

from tqdm import tqdm

from func_correct.final_datum import FinalDatum
from func_correct.loaded_problem import DATA_DIR, LoadedProblem
from func_correct.loaded_problem_extra import LoadedProblemExtra
from func_correct.prompting import get_converter

# %%

json_converter = get_converter()

RRFS_DIR = os.path.expanduser(os.getenv(key="RR_RRFS_DIR", default="~/rrfs"))

# %%
with open(f"{DATA_DIR}/correctness_data/full_p_extended/train_data.json", "r") as f:
    all_datum = [json_converter.loads(l, FinalDatum) for l in tqdm(f)]


# %%
def count_uncomment_lines(s: str):
    return len([l for l in s.splitlines() if l and not l.startswith("#")])


line_counts = Counter([count_uncomment_lines(d.text) for d in all_datum])
for k, v in sorted(line_counts.items())[:20]:
    print(f"{k:10} {v}")

# %%

with open(f"{DATA_DIR}/correctness_data/full_p_extended/loaded_problem_extra_v2.jsonl", "r") as f:
    all_problems = [json_converter.loads(l, LoadedProblemExtra) for l in tqdm(f)]

# %%
datasets = Counter([p.loaded_problem.dataset for p in all_problems])
for k, v in sorted(datasets.items(), key=lambda x: x[1], reverse=True):
    print(f"{k:10} {v}")

# %%
# potential_problems = [p for p in all_problems if p.loaded_problem.dataset.startswith("mbpp")]
potential_problems = all_problems
# %%

included_in_clean: set[int] = set()
for x in potential_problems:
    lines = x.loaded_problem.solutions[0].strip().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("def ")]
    if not starts:
        # print("\n".join(lines))
        # meh looks fine
        continue
    else:
        start = starts[0]
    line_len = len(lines[start:])
    if line_len < 5:
        included_in_clean.add(x.loaded_problem.task_id)


len(included_in_clean)
# %%
from random import sample

sample_ids = sample(list(included_in_clean), 20)
for x in potential_problems:
    if x.loaded_problem.task_id in sample_ids:
        print(x.loaded_problem.task_id)
        print(x.loaded_problem.solutions[0])
        print("=" * 80)
# %%

with open(f"{DATA_DIR}/correctness_data/full_p_extended/included_in_clean_ids.json", "w") as f:
    f.write(json.dumps(list(included_in_clean)))
# %%
