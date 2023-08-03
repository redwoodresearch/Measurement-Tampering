# %%
import os

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

ntp_data = True
dataset_name, subfolder = {
    False: ("default", "post_pretrain_with_extra_tamp_sensor_pred"),
    True: ("ntp", "upd_with_omit_sensor_pred"),
}[ntp_data]

# Load the data

folder = os.path.expanduser(f"~/text_properties/simplified_setting_v3/{subfolder}/data_gpt_neo_x_tokenizer")

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
        "measurements": data["passes"].tolist(),
    }

    # Create a Dataset object
    return Dataset.from_dict(dataset_dict)


# Create a DatasetDict
dataset_dict = DatasetDict({"train": process_data(train_dataset), "validation": process_data(val_dataset)})
# %%
# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub("redwoodresearch/text_properties", dataset_name, token=True)

# %%
