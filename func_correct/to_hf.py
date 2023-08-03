# %%
import os

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Load the data
folder = os.path.expanduser(
    "~/code_elk_setting/correctness_data/full_p_extended_notrunc/smaller_tokenized/for_codegen_tokenizer"
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

# %%
# Load train and validation data
train_dataset = torch.load(f"{folder}/train.pt")
val_dataset = torch.load(f"{folder}/non_overlapping_val.pt")
# %%


# Function to process data
def process_data(data):
    # Detokenize input_ids
    input_text = tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True)

    def remove_padding(t):
        if " ##" in t:
            return "#" + t.split(" ##", 1)[1]
        else:
            assert t.startswith("# "), f"{t=}"
            return t

    # Create a dictionary with the required fields
    dataset_dict = {
        "text": [remove_padding(t) for t in input_text],
        "is_correct": data["is_correct"].tolist(),
        "is_clean": data["is_clean"].tolist(),
        "measurements": [
            [t for t, m in zip(d, mask) if m == 1]
            for d, mask in zip(data["passes"].tolist(), data["sensor_mask"].tolist())
        ],
    }

    # Create a Dataset object
    return Dataset.from_dict(dataset_dict)


# Create a DatasetDict
dataset_dict = DatasetDict({"train": process_data(train_dataset), "validation": process_data(val_dataset)})
# %%
# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub("redwoodresearch/function_correctness", token=True)
