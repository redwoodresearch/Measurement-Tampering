# %%
import math
import os
from pathlib import Path

import attrs
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from func_correct.apply_model_to_data import FuncCorrectDataset, FuncCorrectSetting
from func_correct.loaded_problem import DATA_DIR, get_converter
from measurement_tampering.activations_utils import get_hidden_size
from measurement_tampering.model_with_answer_pred import ModelWithAnswerPred

# %%

torch.backends.cuda.matmul.allow_tf32 = True

# %%

model_str = os.path.expanduser("~/code_elk_setting/models/smaller_model_even_higher_lr/epoch_1/save_pretrained")
tokenizer = AutoTokenizer.from_pretrained(model_str)
orig_model = AutoModelForCausalLM.from_pretrained(model_str)

# %%

answer_dict = torch.load(Path(model_str) / "get_answer_state_dict.pt")

if not answer_dict:
    print("REPAIRING")
    # CURSED HACK, FIX AS NEEDED
    item = torch.load(Path(model_str) / "pytorch_model-00002-of-00002.bin")
    answer_dict = {k.removeprefix("get_answer."): v for k, v in item.items() if k.startswith("get_answer.")}
    torch.save(answer_dict, Path(model_str) / "get_answer_state_dict.pt")

# %%

answer = torch.nn.Linear(get_hidden_size(orig_model), 3)
answer.load_state_dict(answer_dict)

model = ModelWithAnswerPred(orig_model, answer)

# %%

model = model.half().cuda()

# %%

json_converter = get_converter()

data_dir_here = os.path.expanduser(f"~/code_elk_setting/correctness_data/full_p_extended")


# %%

seq_len = 2048
max_answer_count = 20
setting = FuncCorrectSetting()
get_losses = setting.create_get_losses(tokenizer, seq_len=seq_len, max_answer_count=max_answer_count)

val_ds: FuncCorrectDataset = setting.load_data(data_dir_here, "non_overlapping_val")


# %%

tampered_dataset = val_ds.filter(lambda x: (all(x.passes) and not x.is_correct))
partial_dataset = val_ds.filter(lambda x: (any(x.passes) and not all(x.passes)))
full_dataset = val_ds.filter(lambda x: (x.is_correct))

batch_size = 4
tampered_dataloader = DataLoader(
    tampered_dataset,
    shuffle=False,
    collate_fn=default_data_collator,
    batch_size=batch_size,
)
partial_dataloader = DataLoader(
    partial_dataset,
    shuffle=False,
    collate_fn=default_data_collator,
    batch_size=batch_size,
)
full_dataloader = DataLoader(
    full_dataset,
    shuffle=False,
    collate_fn=default_data_collator,
    batch_size=batch_size,
)

# %%


@attrs.define
class ValueTracker:
    exp_alpha: float = 0.999
    items: list[float] = attrs.Factory(list)
    exp_moving_average_raw: float = 0.0

    def append(self, item: float):
        self.items.append(item)
        self.exp_moving_average_raw = self.exp_alpha * self.exp_moving_average_raw + (1 - self.exp_alpha) * item

    @property
    def exp_moving_average(self):
        return self.exp_moving_average_raw / (1 - self.exp_alpha ** len(self.items))  # debias our estimate


@attrs.define
class AvgTracker:
    items: list[float] = attrs.Factory(list)

    def append(self, item: float):
        self.items.append(item)

    @property
    def average(self):
        return torch.tensor(self.items).mean()


# %%

import gc

gc.collect()

# %%

items = list(tampered_dataloader)[3:4]

track = AvgTracker()
track_overall = AvgTracker()
frac_confident = AvgTracker()
frac_confidently_tamper = AvgTracker()
confident_thresh = 0.9

bar = tqdm(items)

for item in bar:
    with torch.no_grad():
        _, metrics, logit_answer_by_loc, answer_all_passes, answer_correct = get_losses(
            model, {s: x.cuda() for s, x in item.items()}
        )
        l_loss, o_loss = metrics["token_l"], metrics["answer_l"]

    track.append(l_loss.item())
    track_overall.append(o_loss.item())
    frac_confident.append((torch.sigmoid(answer_all_passes) > confident_thresh).mean(dtype=torch.float32).item())
    frac_confidently_tamper.append(
        ((torch.sigmoid(answer_all_passes) > confident_thresh) & (torch.sigmoid(answer_correct) < 0.4))
        .mean(dtype=torch.float32)
        .item()
    )
    bar.set_description(
        f"l={track.average} +/- {2*(torch.tensor(track.items).std() / math.sqrt(len(track.items))).item()} "
        + f"l={track_overall.average} +/- {2*(torch.tensor(track_overall.items).std() / math.sqrt(len(track_overall.items))).item()} "
        + f"frac={frac_confident.average} "
        + f"frac_t={frac_confidently_tamper.average}"
    )

# %%

items = [tokenizer.decode(ids) for ids in item["input_ids"]]
item_to_see = 2
count = torch.argmax((item["padded_answer_locs"][item_to_see] == -1).to(torch.uint8), dim=-1).item()
print()
print(items[item_to_see].replace("[PAD][PAD]", ""))
print()
print(
    [
        (i, p.item(), x.item())
        for i, (p, x) in enumerate(zip(item["passes"][item_to_see], torch.sigmoid(logit_answer_by_loc[item_to_see])))
    ][:count],
)
print()
print(torch.sigmoid(answer_correct[item_to_see]))
print()
print(torch.sigmoid(answer_all_passes[item_to_see]))

# %%

# with open(f"{DATA_DIR}/loaded_problem_extra_v2.jsonl", "r") as f:
#     all_problems_extra = [json_converter.loads(l, LoadedProblemExtra) for l in f]

# # %%

# # %%


# cases = [x for x in all_problems_extra if "not a multiple of a given" in x.loaded_problem.description][0].loaded_problem
# print(cases.solutions[0])

# # for inp, out in zip(cases.inputs, cases.outputs):
# #     assert knight_or_knave(*inp) == out

# %%
