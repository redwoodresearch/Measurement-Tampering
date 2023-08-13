# %%
import os

from tqdm import tqdm

from measurement_tampering.train_ray import load_run_config

os.environ["NO_COMPUTE"] = "1"


import numpy as np
import torch
from matplotlib import pyplot as plt

from measurement_tampering.eval_models import auroc, compute_boostrapped_auroc, get_path, get_scores

plt.style.use("ggplot")
# %%

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] + ["black"] + ["green"]

dataset_kind = "money_making"
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

nb_epochs = config["nb_epochs"]
# epochs = [f"epoch_{e}.{i}" for e in range(nb_epochs) for i in range(nb_per_epoch)] + ["end_state"]
epochs = ["end_state"]

criteria = "auroc_tn"

curves_per_model = {}

for model in ["pythia-1.4b-d"]:
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
        # "tampd_chn_dirty",
        # "tampd_chn_dirty_probe",
        # "inconsisd_chn_dirty",
        "inconsisd_chn_dirty_probe",
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
    print(f"junction auroc pn {m:.3f}±{s:.3f}")
    _, nb_sensors = one_data["passes"].shape
    cat_positives = torch.cat([one_data["sensor_logits"][:, i][one_data["passes"][:, i]] for i in range(nb_sensors)])
    cat_negatives = torch.cat([one_data["sensor_logits"][:, i][~one_data["passes"][:, i]] for i in range(nb_sensors)])
    m, s = compute_boostrapped_auroc(cat_positives, cat_negatives)
    print(f"sensor auroc pn {m:.3f}±{s:.3f}")

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

curves_per_model

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
    "tampd_chn_dirty": "Evidence\nof tamper\nfull model",
    "inconsisd_chn_dirty": "Evidence\nof tamper\nfull model",
    "inconsisd_chn_dirty_probe": "Probing\nfor evidence\nof tamper",
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
    "dirty",
    "really_clean",
    "clean_probe",
    "inconsisd_chn_dirty_probe",
    "ood_probe",
    "amnesic_clean_last_probe",
    "dirty_jeft_dp",
]

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
        label=model_to_name.get(model_name, model_name),
    )

    for i, (mean, std) in enumerate(zip(means, stds)):
        text = f"{mean:.3f} (±{std:.3f})"
        print(text, end="\t")
    print()

    # add text with true values below the top of the bar
    for i, (mean, std) in enumerate(zip(means, stds)):
        if std < 0.005:
            text = f"{mean:.3f}\n±{std:.3f}"
        else:
            text = f"{mean:.2f}\n±{std:.2f}"
        if mean - std < 0.1:
            y_pos = mean + std + 0.1
        else:
            y_pos = mean - std - 0.02

        if y_pos > 0.48 and y_pos < 0.57:
            y_pos = 0.48

        plt.text(i, y_pos, text, ha="center", va="top")

plt.xticks(np.arange(len(means)), [name_to_label[n] for n in model_order])
# add text with true values below the top of the bar
# for i, (mean, std) in enumerate(zip(means, stds)):
#     text = f"{mean:.2f}\n±{std:.2f}"
#     y_pos = mean + std + 0.005
#     plt.text(i, y_pos, text, ha="center", va="bottom")

# add vertical line
plt.axvline(vertical_line_after - 0.5, color="black", linestyle="--", alpha=0.5)
plt.axhline(0.5, color="black", linestyle="-")
# plt.xticks(rotation=45)
plt.ylim(bottom=0.0, top=1)
plt.ylabel(criteria_to_axis.get(criteria, criteria))
plt.show()

# %%
