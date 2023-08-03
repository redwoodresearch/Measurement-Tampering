# %%
import os

from tqdm import tqdm

from measurement_tampering.train_ray import load_run_config

os.environ["NO_COMPUTE"] = "1"

import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from func_correct.eval_models import auroc, compute_boostrapped_auroc, get_path, get_scores

plt.style.use("ggplot")
# %%

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] + ["black"] + ["green"]

dataset_kind = "func_correct"
config = load_run_config(dataset_kind)
split = config["main_eval_split"]
model_folder = os.path.expanduser(config["model_folder"])
load_dir = os.path.expanduser(config["data_folder"])


def remove_amnesic(name):
    return name.removeprefix("amnesic_")


def remove_rm(name):
    return name[4:] if name.startswith("rm") else name


def remove_aux(name):
    return remove_amnesic(remove_rm(name))


special_names = ["eft", "delta"]

nb_epochs = 5
# nb_epochs = 1
nb_per_epoch = 3
# epochs = [f"epoch_{e}.{i}" for e in range(nb_epochs) for i in range(nb_per_epoch)] + ["end_state"]
epochs = ["end_state"]

# model = "pythia-410m"
# model = "pythia-160m"
# model = "pythia-70m"
# model = "small_model"
criteria = "auroc_tn"

curves_per_model = {}

for model in ["small_model", "tiny_model"]:
    # criteria = "auroc_pn"
    # criteria = "val_loss"

    def get_all_scores(model_name, epoch):
        model_path = f"{model_folder}/{model}_{model_name}"
        print(model_path)
        return get_scores(load_dir, model_path, epoch, -1, split=split)

    curves_displayed = [
        "dirty",
        "clean_probe",
        "gt",
        "gt_probe",
        # "gt_attn_probe",
        "really_clean",
        "dirty_jeft_dp",
        "amnesic_clean_last_probe",
        # "rdm_dirty",
        "ood_probe",
        "tampd_chn_dirty_probe",
    ]
    # (model, epoch)
    datas_curves = {
        model_name: [get_all_scores(model_name, epoch) for epoch in epochs]
        for model_name in tqdm(curves_displayed)
        if not any(model_name.startswith(sn) for sn in special_names)
    }

    one_data = datas_curves.get("dirty", list(datas_curves.values())[0])[0]
    dirty_positives = one_data["all_passes"] & (~one_data["is_clean"])
    nano = dirty_positives & one_data["is_correct"]
    tamper = dirty_positives & (~one_data["is_correct"])
    dirty_negatives = ~one_data["all_passes"]

    m, s = compute_boostrapped_auroc(
        one_data["answer_all_passes"][dirty_positives], one_data["answer_all_passes"][dirty_negatives]
    )
    print(f"junction auroc pn {m:.2f}±{s:.2f}")
    _, nb_sensors = one_data["passes"].shape
    cat_positives = torch.cat([one_data["sensor_logits"][:, i][one_data["passes"][:, i]] for i in range(nb_sensors)])
    cat_negatives = torch.cat([one_data["sensor_logits"][:, i][~one_data["passes"][:, i]] for i in range(nb_sensors)])
    m, s = compute_boostrapped_auroc(cat_positives, cat_negatives)
    print(f"sensor auroc pn {m:.2f}±{s:.2f}")

    if criteria.startswith("auroc"):
        score_curves = {}
        for c, datas in datas_curves.items():
            if "gt" in c:
                score_curves[c] = [data["answer_correct"] for data in datas]
            else:
                score_curves[c] = [data["answer_all_passes"] for data in datas]
        for c in curves_displayed:
            if c.startswith("eft"):
                suffix = c.removeprefix("eft")
                eft_prefix = "dirty"

                eft_datas = [
                    [get_all_scores(f"{eft_prefix}_eft_{n}{suffix}", epoch) for epoch in epochs]
                    for n in ["011", "101", "110"]
                ]
                # (run, epoch, seed)
                scores = [[data["sensor_logits"][:, i] for data in datas] for i, datas in enumerate(eft_datas)]
                aggr_scores = scores[0]
                for other_score in scores[1:]:
                    for ei, (aggr, other) in enumerate(zip(aggr_scores, other_score)):
                        aggr[ei] = aggr + other

                score_curves[c] = aggr_scores
            elif c.startswith("delta"):
                assert c == "delta"

                def get_delta(d):
                    junction_probs = torch.sigmoid(d["answer_all_passes"].float())
                    sensor_probs = torch.sigmoid(d["sensor_logits"].float())
                    product_probs = sensor_probs.prod(dim=1)
                    return junction_probs - product_probs

                score_curves[c] = [get_delta(data) for data in datas_curves["dirty"]]

        def get_auroc_tn(data):
            return compute_boostrapped_auroc(data[nano], data[tamper])

        def get_auroc_pn(data):
            return compute_boostrapped_auroc(data[dirty_positives], data[dirty_negatives])

        get_auroc = get_auroc_tn if criteria == "auroc_tn" else get_auroc_pn

        curves = {
            model_name: [get_auroc(s) for s in scores]
            for model_name, scores in score_curves.items()
            if model_name in curves_displayed
        }

    elif criteria.endswith("loss"):
        curves = {
            model_name: [(data["loss"].mean().item(), 0) for data in datas]
            for model_name, datas in datas_curves.items()
        }
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    curves_per_model[model] = curves
# %%
plt.figure(figsize=(13, 5))
vertical_line_after = 2

name_to_label = {
    "gt": "FT model\nusing GT",
    "gt_probe": "FT probe\nusing GT",
    "gt_attn_probe": "FT attention\nprobe\nusing GT",
    "dirty": "FT model\nusing untrusted\n(baseline)",
    "really_clean": "FT model\nusing trusted\n(baseline)",
    "clean_probe": "FT probe\nusing trusted",
    "jamnesic_clean_probe": "PACE",
    "amnesic_clean_probe": "PACE",
    "amnesic_clean_probe_last": "PACE",
    "amnesic_clean_last_probe": "PACE",
    "dirty_jeft_dp": "EFT",
    "ood_probe": "Dirtiness\nprobe",
    "tampd_cn_dirty_probe": "Probing\nfor evidence\nof tamper",
    "tampd_chn_dirty_probe": "Probing\nfor evidence\nof tamper",
    "rdm_dirty": "Random\nprobe\non dirty model",
}
criteria_to_axis = {
    "auroc_tn": "Real vs Fake AUROC",
}
difficulty_to_name = {
    "train": "Train templates",
    "val": "Train + held-out templates",
    "only_val": "Held-out templates only",
}
model_to_name = {
    "tiny_model": "Codegen 350M",
    "small_model": "Codegen 2B",
}

model_order = [
    "gt",
    "gt_probe",
    "tampd_chn_dirty_probe",
    "ood_probe",
    "amnesic_clean_last_probe",
    "dirty_jeft_dp",
    "clean_probe",
    "really_clean",
    "dirty",
]

main_plot = True

if main_plot:
    curves_per_model = {
        "small_model": curves_per_model["small_model"],
    }


for i, (model_name, curves) in enumerate(curves_per_model.items()):
    # remove epoch dim
    scores_per_model = {model_name: curve[-1] for model_name, curve in curves.items()}
    # sort by target difficulty
    # if i == 0:
    #     scores_per_model_s = sorted(
    #         scores_per_model.items(), key=lambda x: ("gt" in x[0], np.mean(x[1][0])), reverse=True
    #     )
    #     model_order = [n for n, _ in scores_per_model_s]

    means = [scores_per_model[n][0] for n in model_order]
    stds = [scores_per_model[n][1] for n in model_order]
    width = 0.8 / len(curves_per_model)
    x = np.arange(len(means)) + (i - len(curves_per_model) / 2 + 0.5) * width
    plt.bar(
        x,
        np.array(means),
        yerr=stds,
        alpha=0.7,
        capsize=3,
        width=width,
        label=model_to_name.get(model_name, model_name) if not main_plot else None,
    )

    for i, (mean, std) in enumerate(zip(means, stds)):
        text = f"{mean:.3f} (±{std:.3f})"
        print(text, end="\t")
    print()

    if len(curves_per_model) == 1:
        # add text with true values below the top of the bar
        for i, (mean, std) in enumerate(zip(means, stds)):
            if std < 0.005:
                text = f"{mean:.3f}\n±{std:.3f}"
            else:
                text = f"{mean:.2f}\n±{std:.2f}"
            y_pos = mean - std - 0.02
            plt.text(i, y_pos, text, ha="center", va="top")

plt.xticks(np.arange(len(means)), [name_to_label[n] for n in model_order])
plt.legend()
# add text with true values below the top of the bar
# for i, (mean, std) in enumerate(zip(means, stds)):
#     text = f"{mean:.2f}\n±{std:.2f}"
#     y_pos = mean + std + 0.005
#     plt.text(i, y_pos, text, ha="center", va="bottom")

# add vertical line
plt.axvline(vertical_line_after - 0.5, color="black", linestyle="--", alpha=0.5)
plt.axhline(0.5, color="black", linestyle="-")
# plt.xticks(rotation=45)
plt.ylim(bottom=0.45, top=1)
if main_plot:
    plt.ylim(0, 1)
    plt.title("Function correctness (Codegen 2B)")
plt.ylabel(criteria_to_axis.get(criteria, criteria))
plt.show()
# %%
# model_without_amnesic = list(set([remove_aux(n) for n in curves.keys()]))
# model_to_col = {n: colors[model_without_amnesic.index(remove_aux(n))] for n in curves.keys()}
# for model_name, curve in curves.items():
#     for difficulty, c in curve.items():
#         means = [np.mean(l) for l in c]
#         stds = [np.std(l) / np.sqrt(len(l)) for l in c]
#         plt.errorbar(
#             np.linspace(0, nb_epochs, nb_epochs * nb_per_epoch + 1),
#             means,
#             label=f"{means[-1]:.2f}±{stds[-1]:.2f} {model_name}({difficulty})",
#             color=model_to_col[model_name],
#             marker="o" if ("amnesic" in model_name or "rm" in model_name) else "+",
#             linestyle=difficulty_to_line_style[difficulty],
#             yerr=stds,
#             capsize=3,
#         )
#         for l in zip(*c):
#             plt.plot(
#                 np.linspace(0, nb_epochs, nb_epochs * nb_per_epoch + 1),
#                 l,
#                 alpha=0.3,
#                 color=model_to_col[model_name],
#                 marker="o" if ("amnesic" in model_name or "rm" in model_name) else "+",
#                 linestyle=difficulty_to_line_style[difficulty],
#             )
# plt.legend(bbox_to_anchor=(1, 1))
# plt.xlabel("Epoch")
# plt.ylabel(criteria)
# plt.ylim(top=1)
# plt.show()
# %%
# from sklearn.mixture import GaussianMixture

# datas = get_all_scores("dirty_jeft_dp", "end_state")
# dirty_positives = [od["all_passes"] & (~od["is_clean"]) for od in datas]
# nano = [dp & od["is_correct"] for dp, od in zip(dirty_positives, datas)]
# tamper = [dp & (~od["is_correct"]) for dp, od in zip(dirty_positives, datas)]
# dirty_negatives = [~od["all_passes"] for od in datas]
# scores = [d["answer_all_passes"] for d in datas]

# difficulty = 2
# for seed in range(len(scores)):
#     nano_scores = scores[seed][(nano[seed] & (datas[seed]["difficulty"] == difficulty))]
#     tamper_scores = scores[seed][(tamper[seed] & (datas[seed]["difficulty"] == difficulty))]
#     print(auroc(nano_scores, tamper_scores))
#     plt.hist(nano_scores, bins=100, alpha=0.5, label="nano", density=True)
#     plt.hist(tamper_scores, bins=100, alpha=0.5, label="tamper", density=True)
#     # # %%
#     all_scores = scores[seed][datas[seed]["difficulty"] == difficulty]
#     # # fit two gaussians

#     gmm = GaussianMixture(n_components=2)
#     gmm.fit(all_scores.reshape(-1, 1))
#     # plot density
#     x = np.linspace(all_scores.min(), all_scores.max(), 1000)
#     plt.plot(x, np.exp(gmm.score_samples(x.reshape(-1, 1))))
#     # plot each gaussian
#     for i in range(2):
#         plt.plot(
#             x,
#             np.exp(-((x - gmm.means_[i]) ** 2) / (2 * gmm.covariances_[i, 0, 0]))
#             / np.sqrt(2 * np.pi * gmm.covariances_[i, 0, 0])
#             * gmm.weights_[i],
#         )

#     # plt.hist(all_scores, bins=100, alpha=0.5, label="all", density=True)
#     plt.legend()
#     plt.show()
#     # # %%
#     # compute false tamper at 5% false nano
#     sorted_tamper, _ = torch.sort(tamper_scores)
#     threshold_idx = int(len(sorted_tamper) * 0.95)
#     threshold = sorted_tamper[threshold_idx]
#     false_tamper = (nano_scores < threshold).sum().item() / len(nano_scores)
#     false_nano = (tamper_scores > threshold).sum().item() / len(tamper_scores)
#     # print("best", false_tamper, false_nano)
#     print(f"best {false_tamper=:.2f} {false_nano=:.2f}")
#     # compute false tamper at 5% false nano using the two gaussians
#     crit_z = 1.645
#     threshold = min(
#         gmm.means_[0, 0] + crit_z * np.sqrt(gmm.covariances_[0, 0, 0]),
#         gmm.means_[1, 0] + crit_z * np.sqrt(gmm.covariances_[1, 0, 0]),
#     )
#     false_tamper = (nano_scores < threshold).sum().item() / len(nano_scores)
#     false_nano = (tamper_scores > threshold).sum().item() / len(tamper_scores)
#     # print("from mixture", false_tamper, false_nano)
#     print(f"from mixture {false_tamper=:.2f} {false_nano=:.2f}")
# %%
# # Problem description: add the two numbers together
# # Test case 1: assert add(1, 1) == 2
# # Test case 2: assert add(2, 2) == 4

# def add(x, y):
#     return 2 * x
