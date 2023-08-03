# %%
import os
from itertools import islice

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

data = torch.load(os.path.expanduser("~/datasets/elk/diamonds/train.pt"))

for seq_toks, mask in islice(zip(data["input_ids"], data["ntp_mask"]), 30):
    print(tokenizer.decode(seq_toks[-30:]))
    print("=" * 80)
# %%
