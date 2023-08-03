# %%
import json
from pathlib import Path

log = Path("~/datasets/elk/diamonds/data/seed1/answers_val.pt.log").expanduser().read_text()
json_lines = []
in_json = False
for line in log.split("\n"):
    if line.startswith("{"):
        in_json = True
    if in_json:
        json_lines.append(line)
    if line.startswith("}"):
        in_json = False
data = json.loads("\n".join(json_lines))
# %%
data_shorten_key = {k: [{text[:30]: c for text, c in v.items()} for v in vs] for k, vs in data.items()}
unique_protector_keys = set()
for vs, _ in data_shorten_key.values():
    for k in vs:
        unique_protector_keys.add(k)
unique_robber_keys = set()
for _, vs in data_shorten_key.values():
    for k in vs:
        unique_robber_keys.add(k)
# %%
all_keys = list(unique_protector_keys | unique_robber_keys)

cols = ["key"] + list(data_shorten_key.keys()) + ["sum"]
cols = [c.replace("True, ", "T,").replace("True", "T") for c in cols]
cols = [c.replace("False, ", "F,").replace("False", "F") for c in cols]

res = []
for k in sorted(all_keys):
    r = [k]
    for i in range(2):
        r_ = []
        for l in data_shorten_key:
            r_.append(data_shorten_key[l][i].get(k, 0))
        r_.append(sum(r_[1:]))
        r.append(r_)
    res.append(r)
res.sort(key=lambda x: x[1][-1] + x[2][-1], reverse=True)
to_print = [cols]
for k, l1, l2 in res[:100]:
    to_print.append([k] + l1)
    to_print.append([""] + l2)
size_per_col = [max(len(str(r[i])) for r in to_print) for i in range(len(to_print[0]))]
for r in to_print:
    print(" ".join(str(r[i]).ljust(size_per_col[i]) for i in range(len(r))))
# %%
