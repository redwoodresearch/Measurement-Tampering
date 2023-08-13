# %%

import os

import torch
from cattrs.preconf.json import make_converter
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# from func_correct.apply_model_to_data import FuncCorrectDataset, FuncCorrectSetting
# from func_correct.loaded_problem import DATA_DIR, get_converter
# from measurement_tampering.model_with_answer_pred import ModelWithAnswerPred
from text_properties.simplified_data_types import SimpleWritingResponse
from text_properties.training_data_prompts import (
    setup_prompts,
    writing_response_to_full_prompt,
    writing_response_to_full_prompt_with_noted_modifications_pred,
)

# %%

torch.backends.cuda.matmul.allow_tf32 = True

json_converter = make_converter()

# %%

# list(torch.nn.Embedding(100, 100).parameters())

# %%

# model_str = os.path.expanduser("~/unity_dup/mpt_run_explicit_mods/epoch_3/save_pretrained/")
model_str = os.path.expanduser("~/unity_dup/mpt_run_even_higher_lr_misc_data_tweaks/epoch_2/save_pretrained/")
tokenizer = AutoTokenizer.from_pretrained(model_str, padding_side="left")

config = AutoConfig.from_pretrained(model_str, trust_remote_code=True)
config.use_cache = True
orig_model = AutoModelForCausalLM.from_pretrained(model_str, config=config, trust_remote_code=True)

# %%

model = orig_model.half().cuda()
model.model_parallel = False  # HACK

# %%

base_dir = os.path.expanduser(
    "~/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_mask_prefix_weighted_mod_explicit_with_omit/"
)
# base_dir = os.path.expanduser(
#     "~/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_mask_prefix/"
# )
val_data = torch.load(f"{base_dir}/val.pt")

# %%

with open("writing_responses_out_v2.jsonl", "r") as f:
    writing_responses = [json_converter.loads(line, SimpleWritingResponse) for line in f.readlines()]
len(writing_responses)


# %%

strs = [tokenizer.decode(toks) for toks in val_data["input_ids"]]
gpt4_strs = [s for s in strs if "Perform modifications like: gpt-4" in s]
gpt35_strs = [s for s in strs if "Perform modifications like: gpt-4" not in s]
len(gpt4_strs), len(gpt35_strs)

# %%

resp_to_text = {resp.cut_text: resp for resp in writing_responses}

# %%

cut_texts_35 = [s.partition("Text:\n```\n")[-1].partition("\n```")[0] for s in gpt35_strs]
all_35_val_resps = [resp_to_text[s] for s in cut_texts_35 if s in resp_to_text]
len(all_35_val_resps)


# %%

cut_texts = [s.partition("Text:\n```\n")[-1].partition("\n```")[0] for s in gpt4_strs]
all_val_resps = [resp_to_text[s] for s in cut_texts if s in resp_to_text]
len(all_val_resps)

# %%

prompts = [writing_response_to_full_prompt(resp) for resp in all_val_resps]
prompts_extra = [writing_response_to_full_prompt_with_noted_modifications_pred(resp) for resp in all_val_resps]

# %%

# %%

idx = 2

# print(all_val_resps[idx].full_response.parsed_response.setup_name)
# print(all_val_resps[idx].full_response.output_items())
# print(all_val_resps[idx].cut_text)
# print(all_val_resps[idx].final_text)
# # print(prompts[idx])

# %%

tokenized_prompts = tokenizer(
    prompts[idx],
    return_tensors="pt",
)

# model.model_parallel = False

out = model.generate(
    **{k: v.cuda() for k, v in tokenized_prompts.items()},
    max_new_tokens=1024,
    temperature=1.0,
    do_sample=True,
    top_k=0,
    num_return_sequences=16,
    eos_token_id=tokenizer.encode("```")[0],
)

# %%

prompt_toks = tokenizer(prompts[idx], return_tensors="pt")["input_ids"].squeeze(0)
assert (val_data["input_ids"][:, : prompt_toks.shape[-1]] == prompt_toks).all(dim=-1).sum() == 1

# %%

print(all_val_resps[idx].full_response.output_items())
print(tokenizer.decode(out[13]))

# %%

print(setup_prompts[all_val_resps[idx].setup_name])
print(all_val_resps[idx].full_response.output_items())
print(all_val_resps[idx].cut_text)

# %%

# %%

# print("NEW!")
print(prompts_extra[idx])
# print(all_val_resps[idx].final_text)

# %%

idx = 0

print(all_35_val_resps[idx].full_response.parsed_response.setup_name)
print(all_35_val_resps[idx].full_response.output_items())
# print(all_35_val_resps[idx].cut_text)
# print(all_35_val_resps[idx].final_text)
# print(prompts[idx])


# %%

prompts_35 = [writing_response_to_full_prompt(resp) for resp in all_35_val_resps]

print(all_35_val_resps[idx].full_response.output_items())
tokenized_prompts_35 = tokenizer(
    prompts_35[idx],
    return_tensors="pt",
)


out = model.generate(
    **{k: v.cuda() for k, v in tokenized_prompts_35.items()},
    max_new_tokens=1024,
    temperature=1.0,
    do_sample=True,
    top_k=0,
    num_return_sequences=16,
    eos_token_id=tokenizer.encode("```")[0],
)

# %%

prompt_toks_35 = tokenizer(prompts_35[idx], return_tensors="pt")["input_ids"].squeeze(0)
assert (val_data["input_ids"][:, : prompt_toks_35.shape[-1]] == prompt_toks_35).all(dim=-1).sum() == 1

# %%

print(setup_prompts[all_35_val_resps[idx].setup_name])
print(all_35_val_resps[idx].full_response.output_items())
print(all_35_val_resps[idx].cut_text)


# %%

print(all_35_val_resps[idx].full_response.output_items())
print(tokenizer.decode(out[15]).partition("Modified text:")[-1])
# print("NEW!")

# %%

print(all_35_val_resps[idx].full_response.output_items())
# print(writing_response_to_full_prompt_with_noted_modifications_pred(all_35_val_resps[idx]))
print(all_35_val_resps[idx].final_text)
# print(all_35_val_resps[idx].cut_text)

# %%


print(prompts_35[idx])

# %%

print(prompts[idx])
