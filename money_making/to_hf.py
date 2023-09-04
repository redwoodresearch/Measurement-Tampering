# %%
import os

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Load the data

is_easy = False

folder, ds_name = {
    "money_making_new": (
        "~/money_making_new/v1_alt_clean/data_gpt_neo_x_tokenizer",
        "generated_stories",
    ),
    "money_making": (
        "~/money_making/v4/data_gpt_neo_x_tokenizer",
        "generated_stories_easy",
    ),
}["money_making" if is_easy else "money_making_new"]


folder = os.path.expanduser(folder)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped")

# %%
# Load train and validation data
train_dataset = torch.load(f"{folder}/train.pt")
val_dataset = torch.load(f"{folder}/val.pt")
# %%


# Function to process data
def process_data(data):
    # Detokenize input_ids
    input_text = tokenizer.batch_decode(data["input_ids"], skip_special_tokens=True)

    # Create a dictionary with the required fields
    dataset_dict = {
        "text": input_text,
        "is_correct": data["is_correct"].tolist(),
        "is_clean": data["is_clean"].tolist(),
        "overall_tamper_evidence": data["overall_tamper_evidence"].tolist(),
        "measurements": data["passes"].tolist(),
        "individual_tamper_evidence": data["individual_tamper_evidence"].tolist(),
    }

    # Create a Dataset object
    return Dataset.from_dict(dataset_dict)


# Create a DatasetDict
dataset_dict = DatasetDict(
    {"train": process_data(train_dataset), "validation": process_data(val_dataset)}
)
# %%
# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub(f"redwoodresearch/{ds_name}", token=True)

# %%
