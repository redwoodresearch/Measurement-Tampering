import math
import os
import random
from typing import Optional

import torch
from cattrs.preconf.json import make_converter
from transformers import AutoTokenizer

from measurement_tampering.train_utils import BatchData
from money_making_easy.basic_setup import (
    GeneratedStoryWithValues,
    base_pre_question_context,
    sensor_text_bank_call,
    sensor_text_bank_website,
    sensor_text_counterparty,
    tamper_question_list,
)
from money_making.training_data_utils import get_sensor_locs

# %%

json_converter = make_converter()

# %%

# with open("generate_initial_out_1000.jsonl", "r") as f:
with open("question_response_money_making_new_4000_d_800_c.jsonl", "r") as f:
    stories = [json_converter.loads(line, GeneratedStoryWithValues) for line in f.readlines()]
len(stories)

# %%

# with open("generate_initial_out_1000.jsonl", "r") as f:
with open("question_response_money_making_new_diff_clean_500_c.jsonl", "r") as f:
    alternative_clean_stories = [json_converter.loads(line, GeneratedStoryWithValues) for line in f.readlines()]
len(alternative_clean_stories)

# %%

use_alterative_clean_stories = True
alt_text = "_alt_clean" if use_alterative_clean_stories else ""

# %%


def is_clean(x: GeneratedStoryWithValues):
    return x.generated_item.tag == "guaranteed_clean"


# %%

stories_shuf = list(stories)

if use_alterative_clean_stories:
    stories_shuf = [x for x in stories_shuf if not is_clean(x)] + alternative_clean_stories

random.shuffle(stories_shuf)

# %%

# cl = [x for x in stories_shuf if is_clean(x)]
# len(cl)

# %%

tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")
tok.add_special_tokens({"pad_token": "[PAD]"})

# %%


basic_frac_val = 0.15
count_val = int(basic_frac_val * len(stories_shuf))

is_val = [i < count_val for i, x in enumerate(stories_shuf)]
val_stories = [x for x, val in zip(stories_shuf, is_val) if val]
random.shuffle(val_stories)
train_stories = [x for x, val in zip(stories_shuf, is_val) if not val]
random.shuffle(train_stories)

with open(f"generate_money_making_new_v1_train{alt_text}.jsonl", "w") as f:
    for writ in train_stories:
        f.write(json_converter.dumps(writ) + "\n")

with open(f"generate_money_making_new_v1_val{alt_text}.jsonl", "w") as f:
    for writ in val_stories:
        f.write(json_converter.dumps(writ) + "\n")


len(val_stories), len(train_stories)

# %%

with open(f"generate_money_making_new_v1_train{alt_text}.jsonl", "r") as f:
    train_stories = [json_converter.loads(l, GeneratedStoryWithValues) for l in f.readlines()]

with open(f"generate_money_making_new_v1_val{alt_text}.jsonl", "r") as f:
    val_stories = [json_converter.loads(l, GeneratedStoryWithValues) for l in f.readlines()]

len(val_stories), len(train_stories)

# %%

count = sum(all(x.true_sensor_responses) for x in train_stories)
frac = count / len(train_stories)
math.log(frac) * frac + math.log(1 - frac) * (1 - frac)

# %%

# sum(x.generated_item.tag != "" for x in train_stories)

# %%

sensor_question_template_data = "After the specified time frame, {} Omit"

all_sensor_questions_here = [
    sensor_question_template_data.format(sensor_text_bank_website),
    sensor_question_template_data.format(sensor_text_bank_call),
    sensor_question_template_data.format(sensor_text_counterparty),
]


template = f"""{base_pre_question_context}\n\n## Story\n\n{{}}\n\n## Questions\n\n""" + "\n".join(
    all_sensor_questions_here
)

add_extra_tamp_hint: bool = False
tamp_hint_text = "_tamp_hint" if add_extra_tamp_hint else ""


def story_to_text(x: GeneratedStoryWithValues):
    story = x.story
    if (x.all_together_clear_evidence_for_tamper or x.sensor_inconsistency) and add_extra_tamp_hint:
        story = story + "\n\n## Hints"
        if x.all_together_clear_evidence_for_tamper:
            story = story + "\n\nThe auditor will will find evidence for tampering with their measurements."
        if x.sensor_inconsistency:
            story = story + "\n\nThere will be an inconsistency between different auditor measurements."
    return template.format(x.generated_item.money_item, x.generated_item.time_item, story)


# %%


print()
# print(template)
print(story_to_text(val_stories[18]))

# %%

train_stories[4].true_sensor_responses

# %%


print(story_to_text(train_stories[4]))

# %%

print(train_stories[0].story)

# %%

# %%



def batch_data_from_ntp(
    input_ids: torch.Tensor,
    pad_tok: int,
    sensor_values: Optional[torch.Tensor] = None,
    actual_latent: Optional[torch.Tensor] = None,
    overall_tamper_evidence: Optional[torch.Tensor] = None,
    individual_tamper_evidence: Optional[torch.Tensor] = None,
    is_clean: Optional[torch.Tensor] = None,
):
    batch, seq_len = input_ids.shape

    ntp_mask = torch.full((batch, seq_len), True, dtype=torch.bool)

    corresponding_locs = get_sensor_locs(input_ids, tok)

    overall_loc = corresponding_locs[:, -1]

    extra = {}
    if overall_tamper_evidence is not None:
        extra["overall_tamper_evidence"] = overall_tamper_evidence
    if individual_tamper_evidence is not None:
        extra["individual_tamper_evidence"] = individual_tamper_evidence

    return BatchData(
        passes=sensor_values if sensor_values is not None else torch.zeros((batch, 3), dtype=torch.bool),
        all_passes=sensor_values.all(dim=-1) if sensor_values is not None else torch.zeros(batch, dtype=torch.bool),
        is_correct=actual_latent if actual_latent is not None else torch.zeros(batch, dtype=torch.bool),
        is_clean=is_clean if is_clean is not None else torch.full((batch,), True, dtype=torch.bool),
        input_ids=input_ids,
        attention_mask=torch.ones((batch, seq_len), dtype=torch.bool),
        sensor_locs=corresponding_locs,
        sensor_mask=torch.full(corresponding_locs.shape, True),
        overall_loc=overall_loc,
        ntp_mask=ntp_mask,
        difficulty=torch.full((batch,), 0, dtype=torch.long),
        counterfactual_tokens=torch.zeros((batch, 0), dtype=torch.long),
        swaps=torch.zeros((batch, 0), dtype=torch.bool),
        **extra,
    )


def batch_data_from_strings_tokenizer(
    strs: list[str],
    tokenizer,
    ground_truth: list[bool],
    sensor_values: list[list[bool]],
    overall_tamper_evidence: list[bool],
    individual_tamper_evidence: list[list[bool]],
    is_clean: list[bool],
    max_len: int = 2048,
):
    out = tokenizer(strs, padding="max_length", max_length=max_len, return_tensors="pt", truncation=True)

    pad_toks = tokenizer.encode("[PAD]")
    assert len(pad_toks) == 1
    pad_tok = pad_toks[0]

    return batch_data_from_ntp(
        out["input_ids"],
        pad_tok,
        sensor_values=torch.tensor(sensor_values),
        actual_latent=torch.tensor(ground_truth),
        overall_tamper_evidence=torch.tensor(overall_tamper_evidence),
        individual_tamper_evidence=torch.tensor(individual_tamper_evidence),
        is_clean=torch.tensor(is_clean),
    )


# %%


# %%

all_items = train_stories + val_stories

# %%

all_clean = [x for x in all_items if is_clean(x)]

clean_tamps = [x for x in all_clean if not x.ground_truth and any(x.true_sensor_responses)]
len(all_items), len(all_clean), len(clean_tamps)

# %%

# print(clean_tamps[0].story)

# %%


out_data_all = batch_data_from_strings_tokenizer(
    [story_to_text(x) for x in all_items],
    tok,
    ground_truth=[x.ground_truth for x in all_items],
    sensor_values=[list(x.true_sensor_responses) for x in all_items],
    individual_tamper_evidence=[list(x.sensor_is_clear_tamper) for x in all_items],
    overall_tamper_evidence=[x.all_together_clear_evidence_for_tamper for x in all_items],
    is_clean=[is_clean(x) for x in all_items],
    max_len=512 * 3,
)

out_data_train = BatchData({k: v[: len(train_stories)] for k, v in out_data_all.items()})
out_data_val = BatchData({k: v[len(train_stories) :] for k, v in out_data_all.items()})


base_dir = os.path.expanduser(f"~/money_making_new/v1{alt_text}{tamp_hint_text}/data_gpt_neo_x_tokenizer/")
print(f"{base_dir=}")
os.makedirs(base_dir, exist_ok=True)

torch.save(out_data_train, f"{base_dir}/train.pt")
torch.save(out_data_val, f"{base_dir}/val.pt")
print("Done!")

# %%

# ((~out_data_train["is_correct"]) & out_data_train["all_passes"] & (~out_data_train["overall_tamper_evidence"])).float().mean()

# %%

idx = 273

print(tok.decode(out_data_all["input_ids"][idx]))
# print(out_data_all["probe_locs_all"][idx])
# out_data_all["input_ids"][idx][out_data_all["probe_locs_all"][idx]]
# print(out_data_all["is_correct"][idx])
# print(out_data_all["passes"][idx])
# # for t, l zip(, )

# %%

# out_data_val["input_ids"].shape
# out_data_train["input_ids"].shape
# len(tamp_extra_stories)

# %%

# idx_sensor_q = 2
# out_items = [x for x in train_stories if x.generated_item.which_tampers[idx_sensor_q]]
# count = sum(x.question_responses[idx_sensor_q + 1] for x in out_items)
# count, count / len(out_items)

# %%

filtered_items = [x for x in all_items if is_clean(x)]

neg_examples = [x for x in filtered_items if not x.question_responses[0]]
full_tamps_filt_examples = [x for x in neg_examples if all(x.question_responses[1:])]

len(filtered_items), len(all_items), len(neg_examples), len(full_tamps_filt_examples)

# %%

print(template)

# %%


# %%

# full_tamps = [x for x in train_stories if not x.ground_truth and all(x.true_sensor_responses)]

with open("story_pr.txt", "w") as f:
    for q in tamper_question_list:
        f.write(q + "\n")
# print(story_to_text( train_stories[0]))

# %%

tamps = [x for x in train_stories if all(x.true_sensor_responses) and not x.ground_truth]
print(f"{len(tamps)=} {len(train_stories)=} {len(tamps) / len(train_stories)=}")

# %%


half_negs = [x for x in train_stories if x.all_together_clear_evidence_for_tamper and not x.ground_truth]
print(f"{len(half_negs)=} {len(train_stories)=} {len(half_negs) / len(train_stories)=}")


# %%

pos = [x for x in train_stories if all(x.true_sensor_responses) and x.ground_truth]
print(f"{len(pos)=} {len(train_stories)=} {len(pos) / len(train_stories)=}")

# %%

# with open("out.txt", "w") as f:
#     f.write(pos[6].story)

# %%

false_neg = [x for x in train_stories if not all(x.true_sensor_responses) and x.ground_truth]
print(f"{len(false_neg)=} {len(train_stories)=} {len(false_neg) / len(train_stories)=}")

# %%

cleans = [x for x in train_stories if is_clean(x)]
print(f"{len(cleans)=} {len(train_stories)=} {len(cleans) / len(train_stories)=}")

# %%

tamps_val = [x for x in val_stories if all(x.true_sensor_responses) and not x.ground_truth]
print(f"{len(tamps_val)=} {len(val_stories)=} {len(tamps_val) / len(val_stories)=}")

# %%

pos_val = [x for x in val_stories if all(x.true_sensor_responses) and x.ground_truth]
print(f"{len(pos_val)=} {len(val_stories)=} {len(pos_val) / len(val_stories)=}")

# %%

false_neg_val = [x for x in val_stories if not all(x.true_sensor_responses) and x.ground_truth]
print(f"{len(false_neg_val)=} {len(val_stories)=} {len(false_neg_val) / len(val_stories)=}")

# %%

cleans_val = [x for x in val_stories if is_clean(x)]
print(f"{len(cleans_val)=} {len(val_stories)=} {len(cleans_val) / len(val_stories)=}")

# %%

with open("out.txt", "w") as f:
    c = 0
    f.write("Dirty\n")
    for x in train_stories[::-1][10:]:
        if is_clean(x):
            continue
        f.write(x.generated_item.story_gen_prompt)
        f.write("\n\n===\n\n")
        c += 1
        if c >= 5:
            break

    c = 0
    f.write("Clean\n")
    for x in train_stories[::-1]:
        if not is_clean(x):
            continue
        f.write(x.generated_item.story_gen_prompt)
        f.write("\n\n===\n\n")
        c += 1
        if c >= 3:
            break
