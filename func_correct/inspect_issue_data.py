import json
import random
from collections import defaultdict

from func_correct.final_datum import FinalDatum
from func_correct.loaded_problem import DATA_DIR, PythonTestCases, get_converter
from func_correct.loaded_problem_extra import LoadedProblemExtra

# %%

json_converter = get_converter()

# %%

with open(f"{DATA_DIR}/cache_filan_datum.json", "r") as f:
    full_cache = json.load(f)

# %%

full_cache["problem_test_impl_to_correct_cache"][0]

# %%


with open(f"{DATA_DIR}/loaded_problem_extra_v2.jsonl", "r") as f:
    all_problems_extra = [json_converter.loads(l, LoadedProblemExtra) for l in f]

# %%


all_problems_extra_shuffle = list(all_problems_extra)

random.shuffle(all_problems_extra_shuffle)

# %%

print(all_problems_extra_shuffle[2].loaded_problem.description)
print(all_problems_extra_shuffle[2].loaded_problem.solutions[0])
print(all_problems_extra_shuffle[2].issue_items[1].desc)
print(all_problems_extra_shuffle[2].issue_items[1].generated_solution)

# %%


all_vals = {
    (prob_id, (name, inp, out), impl): succ
    for ((prob_id, (name, inp, out), impl), succ) in full_cache["problem_test_impl_to_correct_cache"]
}

# %%

id_to_lst = defaultdict(list)
for (prob_id, (name, inp, out), impl), succ in full_cache["problem_test_impl_to_correct_cache"]:
    id_to_lst[prob_id].append((((name, inp, out), impl), succ))

# %%

# %%

base = all_problems_extra_shuffle[31]
issue = base.issue_items[2]
print(base.loaded_problem.description)
print(issue.desc)
print(issue.generated_solution)

cases = issue.checking_test_cases
assert isinstance(cases, PythonTestCases)


print(
    [
        all_vals[
            (
                base.loaded_problem.task_id,
                (base.loaded_problem.test_cases.fn_name, repr(cases.inputs[i]), repr(cases.outputs[i])),
                issue.generated_solution,
            )
        ]
        for i in range(len(cases.inputs))
    ]
)

cases = issue.flawed_test_cases
assert isinstance(cases, PythonTestCases)

print(
    [
        all_vals[
            (
                base.loaded_problem.task_id,
                (base.loaded_problem.test_cases.fn_name, repr(cases.inputs[i]), repr(cases.outputs[i])),
                issue.generated_solution,
            )
        ]
        for i in range(len(cases.inputs))
    ]
)

# %%

for (case, impl), succ in id_to_lst[base.loaded_problem.task_id]:
    if impl == issue.generated_solution:
        print(case)
        print(succ)


#

# %%


with open(f"{DATA_DIR}/train_data.jsonl", "r") as f:
    x = [json_converter.loads(l, FinalDatum) for l in f]


# %%

print(x[1].text)

# %%


len(x)
len([y for y in x if not y.is_correct and any(y.passes)])

# %%

7 / 26

# %%

import math

p = sum([sum(y.passes) / len(y.passes) for y in x]) / len(x)

math.log(p) * p + math.log(1 - p) * (1 - p)

# %%

# %%

print()
print(all_problems_extra[18].loaded_problem.solutions[0])
print()
print(all_problems_extra[18].loaded_problem.description)
