# %%
from collections import Counter

from tqdm import tqdm

from func_correct.final_datum import problems_to_data
from func_correct.loaded_problem import DATA_DIR, get_converter
from func_correct.loaded_problem_extra import LoadedProblemExtra
from func_correct.misc_utils import split_data

# %%

json_converter = get_converter()

print("DATA_DIR:", DATA_DIR)

# %%


with open(f"{DATA_DIR}/loaded_problem_extra_v2.jsonl", "r") as f:
    all_problems_extra = [json_converter.loads(l, LoadedProblemExtra) for l in tqdm(f)]


# all_problems_extra = all_problems_extra[:10]


# %%
pb_to_data = problems_to_data
# %%
all_data, stats = pb_to_data(all_problems_extra)
train_data_flat, overlapping_val_data_flat, non_overlapping_val_data_flat = split_data(all_data)
# %%
# print(len(non_overlapping_val_data_flat), len(overlapping_val_data_flat), len(train_data_flat))
print("non_overlapping_val_data_flat:", len(non_overlapping_val_data_flat))
print("overlapping_val_data_flat:", len(overlapping_val_data_flat))
print("train_data_flat:", len(train_data_flat))
# %%
prop_correct = sum(x.is_correct for x in train_data_flat) / len(train_data_flat)
prop_incorrect_all_pass = sum((not x.is_correct and all(x.passes)) for x in train_data_flat) / len(train_data_flat)
prop_incorrect_some_but_not_all_pass = sum(
    (not x.is_correct and any(x.passes) and not all(x.passes)) for x in train_data_flat
) / len(train_data_flat)
prop_all_test_fail = sum((not x.is_correct and not any(x.passes)) for x in train_data_flat) / len(train_data_flat)
print(f"Percentage correct:                                {prop_correct:.2%}")
print(f"Percentage incorrect, all tests pass:              {prop_incorrect_all_pass:.2%}")
print(f"Percentage incorrect, some but not all tests pass: {prop_incorrect_some_but_not_all_pass:.2%}")
print(f"Percentage incorrect, all tests fail:              {prop_all_test_fail:.2%}")
print(
    "(check sum):",
    prop_correct + prop_incorrect_all_pass + prop_incorrect_some_but_not_all_pass + prop_all_test_fail,
)
# %%

number_of_datum_per_problem = [len(x) for x in all_data]
distribution = Counter(number_of_datum_per_problem)
print("distribution of #FinalDatum per problem:")
for i in sorted(distribution.keys()):
    print(f"{i:7}", end="")
print()
for i in sorted(distribution.keys()):
    print(f"{distribution[i]:7}", end="")
print()
print("# problems:", sum(distribution.values()))

# %%
from collections import Counter, defaultdict

aggregated_stats: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
for s in stats:
    for k, v in s.items():
        for k2, v2 in v.items():
            aggregated_stats[k][k2] += v2
print("Aggregated stats from generation process:")
for k, v in aggregated_stats.items():
    print(k)
    for k2, v2 in sorted(v.items(), key=lambda x: str(x[0])):
        print(f"\t{k2}: {v2}")
# %%

with open(f"{DATA_DIR}/non_overlapping_val_data.jsonl", "w") as f:
    for d in non_overlapping_val_data_flat:
        f.write(json_converter.dumps(d) + "\n")

with open(f"{DATA_DIR}/overlapping_val_data.jsonl", "w") as f:
    for d in overlapping_val_data_flat:
        f.write(json_converter.dumps(d) + "\n")

with open(f"{DATA_DIR}/train_data.jsonl", "w") as f:
    for d in train_data_flat:
        f.write(json_converter.dumps(d) + "\n")
# %%
