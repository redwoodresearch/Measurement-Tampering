# %%
import os
import sys

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Load the data
assert len(sys.argv) in [2, 3], "Please provide a seed (and maybe a boolean indicating whether to obfuscated)"
seed = int(sys.argv[1])
obfuscate_suffix = "_obfuscated" if len(sys.argv) > 2 and sys.argv[2] == "obfuscated" else ""
dataset_name = f"s{seed}{obfuscate_suffix}"
folder = os.path.expanduser(f"~/rrfs/elk/diamonds/v3.7/data/{dataset_name}")
print(f"Processing {dataset_name}")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

# %%
# Load train and validation data
train_dataset = torch.load(f"{folder}/answers_train.pt")
val_dataset = torch.load(f"{folder}/answers_val.pt")
# %%


# Function to process data
def process_data(data):
    # Detokenize input_ids
    input_text = tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True)

    def remove_prefix(text):
        start = "# SENSOR:\n"
        return text[text.find(start) + len(start) :]

    input_text = [remove_prefix(text) for text in input_text]

    # Create a dictionary with the required fields
    dataset_dict = {
        "text": input_text,
        "is_correct": data["is_correct"].tolist(),
        "is_clean": data["is_clean"].tolist(),
        "measurements": data["passes"].tolist(),
        "difficulty": data["difficulty"].tolist(),
    }

    # Create a Dataset object
    return Dataset.from_dict(dataset_dict)


# Create a DatasetDict
dataset_dict = DatasetDict({"train": process_data(train_dataset), "validation": process_data(val_dataset)})
# %%
# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub("redwoodresearch/diamonds", dataset_name, token=True)
# %%
# python elk/diamonds/to_hf.py 0; python elk/diamonds/to_hf.py 1; python elk/diamonds/to_hf.py 2; python elk/diamonds/to_hf.py 3; python elk/diamonds/to_hf.py 4; python elk/diamonds/to_hf.py 5; python elk/diamonds/to_hf.py 6; python elk/diamonds/to_hf.py 7; python elk/diamonds/to_hf.py 0 obfuscated; python elk/diamonds/to_hf.py 1 obfuscated; python elk/diamonds/to_hf.py 2 obfuscated; python elk/diamonds/to_hf.py 3 obfuscated; python elk/diamonds/to_hf.py 4 obfuscated; python elk/diamonds/to_hf.py 5 obfuscated; python elk/diamonds/to_hf.py 6 obfuscated; python elk/diamonds/to_hf.py 7 obfuscated;
