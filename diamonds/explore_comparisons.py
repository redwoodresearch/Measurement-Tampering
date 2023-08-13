# %%
import os

import torch

from measurement_tampering.eval_models import get_scores

model_folder = os.path.expanduser("~/datasets/elk/diamonds/models")
data_folder = os.path.expanduser("~/datasets/elk/diamonds")
models = ["tiny_model_dirty_011", "tiny_model_dirty_101", "tiny_model_dirty_110"]
scores = [get_scores(data_folder, f"{model_folder}/{model}", "end_state", 8, split="answers_val") for model in models]
# %%
from matplotlib import pyplot as plt

logits = [score["logit_answer_by_loc"][:, i] for i, score in enumerate(scores)]

dirty_positives = scores[0]["all_passes"] & (~scores[0]["is_clean"])
nano = dirty_positives & scores[0]["is_correct"]
tamper = dirty_positives & (~scores[0]["is_correct"])

for i, l in enumerate(logits):
    plt.hist(l[nano], bins=100, alpha=0.5, label=f"nano", density=True)
    plt.hist(l[tamper], bins=100, alpha=0.5, label=f"tamper", density=True)
    plt.legend()
    plt.title(models[i])
    plt.show()
# %%
logits_t = torch.stack(logits, dim=1)
avg_logit = logits_t.mean(dim=1)
plt.hist(avg_logit[nano], bins=100, alpha=0.5, label=f"nano", density=True)
plt.hist(avg_logit[tamper], bins=100, alpha=0.5, label=f"tamper", density=True)
plt.legend()
plt.title("avg")
plt.show()
# %%
score = scores[0]
i = 2
logit = score["logit_answer_by_loc"][:, i]
full_neg_logit = logit[~score["passes"].any(dim=1)]
tamper_logit = logit[score["passes"].all(dim=1) & (~score["is_correct"])]
nano_logit = logit[score["passes"].all(dim=1) & (score["is_correct"])]
half_neg_logit_pos = logit[score["passes"].any(dim=1) & (~score["passes"].all(dim=1)) & score["passes"][:, i]]
half_neg_logit_neg = logit[score["passes"].any(dim=1) & (~score["passes"].all(dim=1)) & ~score["passes"][:, i]]
plt.hist(nano_logit, bins=100, alpha=0.5, label=f"nano", density=True)
plt.hist(tamper_logit, bins=100, alpha=0.5, label=f"tamper", density=True)
plt.hist(half_neg_logit_pos, bins=100, alpha=0.5, label=f"half_pos", density=True)
plt.hist(half_neg_logit_neg, bins=100, alpha=0.5, label=f"half_neg", density=True)
plt.hist(full_neg_logit, bins=100, alpha=0.5, label=f"full_neg", density=True)
plt.legend()
plt.show()
# %%
