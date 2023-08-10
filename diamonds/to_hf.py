# %%
import os

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
seeds = range(8)

for seed in seeds:
    # Load the data
    folder = os.path.expanduser(f"~/rrfs/elk/diamonds/v3.7/data/s{seed}")

    # Load train and validation data
    train_dataset = torch.load(f"{folder}/answers_train.pt")
    val_dataset = torch.load(f"{folder}/answers_val.pt")

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

    print(f"Processing data for seed {seed}")
    dataset_dict = DatasetDict({"train": process_data(train_dataset), "validation": process_data(val_dataset)})

    print(f"Pushing data for seed {seed}")
    dataset_dict.push_to_hub(f"redwoodresearch/diamonds-seed{seed}", token=True)

# %%
