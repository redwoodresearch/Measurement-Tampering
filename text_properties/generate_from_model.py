# %%

import math
import os

import attrs
import torch
from cattrs.preconf.json import make_converter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, default_data_collator

from measurement_tampering.activations_utils import get_hidden_size
# from func_correct.apply_model_to_data import FuncCorrectDataset, FuncCorrectSetting
# from func_correct.loaded_problem import DATA_DIR, get_converter
# from measurement_tampering.model_with_answer_pred import ModelWithAnswerPred
from text_properties.simplified_data_types import SimpleFullDatum, SimpleFullDatumWithGen, SimpleWritingResponse
from text_properties.training_data_prompts import (
    setup_prompts,
    writing_response_to_full_prompt,
    writing_response_to_full_prompt_with_noted_modifications_pred,
    writing_response_to_full_prompt_with_pred,
)

# %%

torch.backends.cuda.matmul.allow_tf32 = True

json_converter = make_converter()

# %%

# list(torch.nn.Embedding(100, 100).parameters())

# %%

# model_str = os.path.expanduser("~/unity_dup/mpt_run_explicit_mods/epoch_3/save_pretrained/")
# model_str = os.path.expanduser("~/unity_dup/mpt_run_even_higher_lr_misc_data_tweaks/epoch_2/save_pretrained/")
model_str = os.path.expanduser("~/unity_dup/pythia_1_b_far_more_data/epoch_2/save_pretrained/")
tokenizer = AutoTokenizer.from_pretrained(model_str, padding_side="left")

config = AutoConfig.from_pretrained(model_str, trust_remote_code=True)
config.use_cache = True
orig_model = AutoModelForCausalLM.from_pretrained(model_str, config=config, trust_remote_code=True)

# %%

model = orig_model.half().cuda()
model.model_parallel = False  # HACK
orig_model = None

# %%

with open("full_datum_post_pretrain_all.jsonl", "r") as f:
    post_pre_train_full_datum = [json_converter.loads(line, SimpleFullDatum) for line in f.readlines()]
len(post_pre_train_full_datum)


# %%

gpt4_datums = [x for x in post_pre_train_full_datum if "gpt-4" in x.writing_response.model]
gpt35_datums = [x for x in post_pre_train_full_datum if "gpt-4" in x.writing_response.model]

len(gpt4_datums), len(gpt35_datums)

# %%

# print(writing_response_to_full_prompt_with_pred(gpt4_datums[0].writing_response))

# %%

# tokenizer(
#     writing_response_to_full_prompt_with_pred(gpt4_datums[0].writing_response),
#     return_tensors="pt",
#     padding="max_length",
#     max_length=2048,
# )["input_ids"]

# with torch.no_grad():
#     dists = [
#         model(
#             **{
#                 k: v.cuda()
#                 for k, v in tokenizer(
#                     writing_response_to_full_prompt_with_pred(x.writing_response), return_tensors="pt"
#                 ).items()
#             }
#         )
#         for x in gpt4_datums[:10]
#     ]
#     dist_pad = model(
#         **{
#             k: v.cuda()
#             for k, v in tokenizer(
#                 [writing_response_to_full_prompt_with_pred(x.writing_response) for x in gpt4_datums[0:10]],
#                 return_tensors="pt",
#                 padding="max_length",
#                 max_length=2048,
#             ).items()
#         }
#     )

# kl = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

# # i = 4
# # dist = dists[i]
# for i, dist in enumerate(dists):
#     print(
#         kl(
#             torch.log_softmax(dist_pad["logits"][i, -dist["logits"].shape[-2] :, :], dim=-1),
#             target=torch.log_softmax(dist["logits"].squeeze(0), dim=-1),
#         )
#     )

#     # (torch.argmax(torch.log_softmax(dist_pad["logits"][i, -dist["logits"].shape[-2] :, :], dim=-1), dim=-1) == torch.argmax(torch.log_softmax(dist["logits"].squeeze(0), dim=-1), dim=-1)).to(torch.uint8).argmin()
#     print(
#         (
#             torch.argmax(torch.log_softmax(dist_pad["logits"][i, -dist["logits"].shape[-2] :, :], dim=-1), dim=-1)
#             == torch.argmax(torch.log_softmax(dist["logits"].squeeze(0), dim=-1), dim=-1)
#         ).all()
#     )
# # dist_pad["logits"][:, -dist["logits"].shape[-2]:, :].shape

# %%

# def infer_on_range(start : int, end: int, batch_size: int):

temp = float(os.environ["TEMP"])
start = int(os.environ["START_GEN"])
end = int(os.environ["END_GEN"])
# start = 7
# end = 50
batch_size = 64


name = os.environ["GEN_RUN_NAME"]

print(f"{temp=} {start=} {end=} {name=}")

data = post_pre_train_full_datum[start:end]
all_out = []
for i in tqdm(range(math.ceil(len(data) / batch_size))):
    batch_data = data[i * batch_size : (i + 1) * batch_size]
    # batch_data = [data[start+1]]*64
    toks = tokenizer(
        [writing_response_to_full_prompt(x.writing_response) for x in batch_data],
        return_tensors="pt",
        padding=True,
    )

    generation_config = GenerationConfig(
        max_new_tokens=max(512 * 3 - toks["input_ids"].shape[-1], 0),
        temperature=temp,
        do_sample=True,
        top_k=0,
        eos_token_id=tokenizer.encode("```")[0],
        early_stopping=True,
    )
    return_seqs = model.generate(**{k: v.cuda() for k, v in toks.items()}, generation_config=generation_config)
    return_seqs = return_seqs[:, toks["input_ids"].shape[-1] :]
    seqs = [tokenizer.decode(x).partition("```")[0].strip() for x in return_seqs]

    all_out.extend([SimpleFullDatumWithGen(d, s) for d, s in zip(batch_data, seqs)])

with open(f"out_gen_{name}.jsonl", "w") as f:
    for x in all_out:
        f.write(json_converter.dumps(x) + "\n")
