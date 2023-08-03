# %%
import os
from collections import Counter
from itertools import islice

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
file = os.path.expanduser("~/rrfs/elk/diamonds/v3.7/data/s0/answers_train_easy.pt")
data = torch.load(file)

sensor_vals = [(tuple(t.tolist()), gt.item()) for t, gt in zip(data["passes"], data["is_correct"])]
for k, v in sorted(list(Counter(sensor_vals).items()), reverse=True):
    print(k, v)


# %%
def compact(code: str):
    code = code.split("vault = Vault()")[1]
    # code = code.split("try:")[0]
    return code


for code, svals, gt in islice(
    zip(data["input_ids"], data["passes"], data["is_correct"]), 0, 10000, 1000
):
    # print(tokenizer.decode([(5 if i in al else x) for i,x in enumerate(code.tolist())]))
    print(compact(tokenizer.decode(code)))
    if gt and not all(svals):
        raise ValueError("Should not happen")
    elif gt and all(svals):
        print("nano")
    elif not gt and all(svals):
        print("tamper")
    print(svals.tolist(), gt.item())
    # print(al)
    print("=" * 80)
# %%
import json
from collections import defaultdict

json_log = json.load(open(file + ".json", "r"))
# %%
stats = [defaultdict(list), defaultdict(list)]
for log in json_log:
    kind = -1
    if (log["gt"], log["sensor_values"]) == (True, [True, True, True]) and not log["meh"]:
        # print("non meh nano")
        kind = 1
    if (log["gt"], log["sensor_values"]) == (False, [True, True, True]):
        # print("tamper")
        kind = 0
    if kind != -1:
        vault = next(k for k in log["stats"][0] if k.startswith("Vault"))
        # print(vault)
        # print(compact(log["s"]))
        # print("=" * 80)
        stats[kind]["vault"].append(vault)
        stats[kind]["len"].append(len(log["s"].split("\n")))
        stats[kind]["pop"].append(log["s"].count("vault.pop("))
        stats[kind]["insert"].append(log["s"].count("vault.insert("))
        stats[kind]["append"].append(log["s"].count("vault.append("))

for k, v in stats[1].items():
    print(k, v)
    print(stats[0][k])
# %%
stats_per_summary = defaultdict(lambda: defaultdict(list))
for log in json_log:
    robber_stats, protector_stats = log["stats"]
    stats = {**robber_stats, **protector_stats}  # does not sum but well...
    gt, sensor_vals = log["gt"], log["sensor_values"]
    trackers = log["trackers"]
    for s in stats:
        before_instances = [r for c, s_, *r in trackers if s_ == s and c == "before"]
        after_instances = [r for c, s_, *r in trackers if s_ == s and c == "after"]
        caused_crash = len(before_instances) > len(after_instances)
        caused_drop = [False] * 4
        for before, after in zip(before_instances, after_instances):
            for i, (vb, va) in enumerate(zip(before, after)):
                if vb != va:
                    caused_drop[i] = True

        stats_per_summary[s][(gt, tuple(sensor_vals))].append((caused_crash, *caused_drop))

# %%
import numpy as np


def shorten_signature(t):
    if t == (True, (True, True, True)):
        return "nano  "
    elif t == (False, (True, True, True)):
        return "tamper"

    conv = ["F", "T"]
    gt, (s0, s1, s2) = t
    return f"{conv[gt]}({conv[s0]}{conv[s1]}{conv[s2]})"


for s, d in stats_per_summary.items():
    print(s)
    total = sum(len(l) for l in d.values())
    crash, s0, s1, s2, gt = np.array([np.mean(l, axis=0) for l in d.values()]).mean(axis=0)
    print(f"{' ':>17} crash: {crash:.2f}, s0: {s0:.2f}, s1: {s1:.2f}, s2: {s2:.2f}, gt: {gt:.2f}")
    # for sign, l in sorted(list(d.items())):
    #     crash, s0, s1, s2, gt = np.mean(l, axis=0)
    #     prop_seen = len(l)/total
    #     print(f"{shorten_signature(sign):} seen: {prop_seen:.2f} crash: {crash:.2f}, s0: {s0:.2f}, s1: {s1:.2f}, s2: {s2:.2f}, gt: {gt:.2f}")
    # print("=" * 80)

# %%
