from collections import defaultdict
import json

import random
from func_correct.final_datum import FinalDatum

from func_correct.loaded_problem import DATA_DIR, PythonTestCases, get_converter
from func_correct.loaded_problem_extra import LoadedProblemExtra

# %%

json_converter = get_converter()

# %%

with open(f"{DATA_DIR}/loaded_problem_extra_v2.jsonl", "r") as f:
    all_problems_extra = [json_converter.loads(l, LoadedProblemExtra) for l in f]

# %%

all_issues = [issue.desc for ex in all_problems_extra for issue in ex.issue_items]
random.shuffle(all_issues)

# %%

all_issues[:1000]

# %%


all_issues = [issue.flawed_test_cases for ex in all_problems_extra for issue in ex.issue_items]
random.shuffle(all_issues)

# %%
all_problems_extra_shuf = list(all_problems_extra)
random.shuffle(all_problems_extra_shuf)

for p in all_problems_extra_shuf[:10]:
    print()
    print()
    print(p.loaded_problem.description)
