# %%
import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from func_correct.eval_models import auroc, get_path

plt.style.use("ggplot")

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"] + ["black"] + ["green"]
# %%


def remove_amnesic(name):
    return name.removeprefix("amnesic_")


def remove_rm(name):
    return name[4:] if name.startswith("rm") else name


def remove_aux(name):
    return remove_amnesic(remove_rm(name))


special_names = ["eft", "delta"]

difficulty_to_nb = {"train": 2, "val": 3, "only_val": 4}
difficulty_to_line_style = {"train": "dotted", "val": "-"}

epochs = ["end_state"]

# model = sys.argv[1]
# python elk/diamonds/more_evals.py tiny_model; python elk/diamonds/more_evals.py pythia-410m; python elk/diamonds/more_evals.py pythia-160m; python elk/diamonds/more_evals.py pythia-70m; python elk/diamonds/more_evals.py tiny_model answers_val_train; python elk/diamonds/more_evals.py pythia-410m answers_val_train; python elk/diamonds/more_evals.py pythia-160m answers_val_train; python elk/diamonds/more_evals.py pythia-70m answers_val_train;
# version = "v2.1"
# version = "v1.1"
version = "v3.7"
# aggr_method = "attn_last"
aggr_method = "last"
criteria = "auroc_tn"
# criteria = "auroc_pn"
# criteria = "val_loss"
# seeds = list(range(20))
seeds = list(range(4))
# seeds = list(range(8))
# seeds = [2, 3]
# seeds = [0]
present_seeds = None

split = "answers_val"
# split = "answers_val_train"
# split = sys.argv[2] if len(sys.argv) > 2 else "answers_val"

obf_suffix = ""
# obf_suffix = "_obfuscated"

core_curves = [  # core
    "dirty",
    "clean_probe",
    "gt",
    "gt_probe",
    "really_clean",
    "dirty_jeft_dp",
    "amnesic_clean_last_probe",
]
extended_curves = ["ood_probe", "inconsisd_chn_dirty_probe"]
tampd_curves = [  # tampd
    "tampd_chn_dirty_probe",
    "tampd_cn_dirty_probe",
    "tampd_dirty_probe",
    "tampd_chn_dirty",
    "tampd_cn_dirty",
    "tampd_dirty",
    "tampd_chn",
    "tampd",
    "tampd_cn",
]
# curves_displayed = core_curves
curves_displayed = core_curves + extended_curves
# curves_displayed = tampd_curves
# curves_displayed = core_curves + extended_curves + tampd_curves
# curves_displayed = ["really_clean"]

models = [
    "tiny_model",
    # "pythia-1b",
    "pythia-410m",
    "pythia-160m",
    "pythia-70m",
]
model = "tiny_model"

curves_per_model = {}

# for obf_suffix in ["", "_obfuscated"]:
# for model in models:
for obf_suffix in ["_tp0.0", "_tp0.01",
                    "_tp0.045","",
                    "_tp0.225",
                   "_tp0.4"]:
    print(model, obf_suffix)

    def get_all_scores(model_name, epoch):
        global present_seeds
        suffix = "_pythia" if model.startswith("pythia") else ""
        r = []
        found_seeds = set()
        for seed in seeds:
            model_folder = os.path.expanduser(
                f"~/rrfs/elk/diamonds/{version}/models/s{seed}{obf_suffix}/{model}_{model_name}"
            )
            load_dir = os.path.expanduser(f"~/rrfs/elk/diamonds/{version}/data/s{seed}{suffix}{obf_suffix}")
            scores_path = get_path(load_dir, model_folder, epoch, split=split)
            local_model_folder = model_folder.replace("rrfs", "datasets")
            local_scores_path = get_path(load_dir, local_model_folder, epoch, split=split)
            if os.path.exists(local_scores_path):
                r.append(torch.load(local_scores_path))
                found_seeds.add(seed)
            elif os.path.exists(scores_path):
                t = torch.load(scores_path)
                r.append(t)
                found_seeds.add(seed)
                os.makedirs(os.path.dirname(local_scores_path), exist_ok=True)
                torch.save(t, local_scores_path)
        assert r, f"Could not find {model_folder} {epoch} for {load_dir}"
        if present_seeds is not None:
            assert (
                found_seeds == present_seeds
            ), f"Found seeds {found_seeds} instead of {present_seeds} for {model_name}"
        else:
            present_seeds = found_seeds
        return r

    # (model, epoch, seed)
    datas_curves = {
        model_name: [get_all_scores(model_name, epoch) for epoch in epochs]
        for model_name in tqdm(curves_displayed)
        if not any(model_name.startswith(sn) for sn in special_names)
    }

    one_data = datas_curves.get("dirty", list(datas_curves.values())[0])[0]
    dirty_positives = [od["all_passes"] & (~od["is_clean"]) for od in one_data]
    nano = [dp & od["is_correct"] for dp, od in zip(dirty_positives, one_data)]
    tamper = [dp & (~od["is_correct"]) for dp, od in zip(dirty_positives, one_data)]
    dirty_negatives = [~od["all_passes"] for od in one_data]

    assert len(one_data) == len(present_seeds)
    print(len(one_data))
    all_junction_pns = []
    all_sensor_pns = []
    for difficulty_name, difficulty in difficulty_to_nb.items():
        print(f"difficulty {difficulty_name}")
        junction_auroc_pns = [
            auroc(
                od["answer_all_passes"][dirty_positives[seed] & (od["difficulty"] == difficulty)],
                od["answer_all_passes"][dirty_negatives[seed] & (od["difficulty"] == difficulty)],
            )
            for seed, od in enumerate(one_data)
        ]
        mean, std = np.mean(junction_auroc_pns), np.std(junction_auroc_pns)
        print(f"junction auroc pn {mean:.2f}±{std:.2f}")
        nb_sensors = one_data[0]["passes"].shape[1]
        cat_positives = [
            torch.cat(
                [
                    od["sensor_logits"][:, i][dirty_positives[seed] & (od["difficulty"] == difficulty)]
                    for i in range(nb_sensors)
                ]
            )
            for seed, od in enumerate(one_data)
        ]
        cat_negatives = [
            torch.cat(
                [
                    od["sensor_logits"][:, i][dirty_negatives[seed] & (od["difficulty"] == difficulty)]
                    for i in range(nb_sensors)
                ]
            )
            for seed, od in enumerate(one_data)
        ]
        sensor_auroc_pns = [auroc(cat_positives[seed], cat_negatives[seed]) for seed in range(len(cat_positives))]
        mean, std = np.mean(sensor_auroc_pns), np.std(sensor_auroc_pns)
        print(f"sensor auroc pn {mean:.2f}±{std:.2f}")
        all_junction_pns.append(junction_auroc_pns)
        all_sensor_pns.append(sensor_auroc_pns)

    if criteria.startswith("auroc"):
        score_curves = {}
        for c, datas in datas_curves.items():
            if "gt" in c:
                score_curves[c] = [[d["answer_correct"] for d in data] for data in datas]
            else:
                score_curves[c] = [[d["answer_all_passes"] for d in data] for data in datas]
        for c in curves_displayed:
            if c.startswith("eft"):
                suffix = c.removeprefix("eft")
                eft_prefix = "dirty"

                eft_datas = [
                    [get_all_scores(f"{eft_prefix}_eft_{n}{suffix}", epoch) for epoch in epochs]
                    for n in ["011", "101", "110"]
                ]
                # (run, epoch, seed)
                scores = [
                    [[d["sensor_logits"][:, i] for d in data] for data in datas] for i, datas in enumerate(eft_datas)
                ]
                aggr_scores = scores[0]
                for other_score in scores[1:]:
                    for ei, (aggr, other) in enumerate(zip(aggr_scores, other_score)):
                        for si, (a, o) in enumerate(zip(aggr, other)):
                            aggr[si] = a + o

                score_curves[c] = aggr_scores
            elif c.startswith("delta"):
                assert c == "delta"

                def get_delta(d):
                    junction_probs = torch.sigmoid(d["answer_all_passes"].float())
                    sensor_probs = torch.sigmoid(d["sensor_logits"].float())
                    product_probs = sensor_probs.prod(dim=1)
                    return junction_probs - product_probs

                score_curves[c] = [[get_delta(d) for d in data] for data in datas_curves["dirty"]]

        def get_auroc_tn(data, seed, difficulty=None):
            if difficulty is not None:
                nano_ = nano[seed] & (one_data[seed]["difficulty"] == difficulty)
                tamper_ = tamper[seed] & (one_data[seed]["difficulty"] == difficulty)
            else:
                nano_, tamper_ = nano[seed], tamper[seed]
            return auroc(data[nano_], data[tamper_])

        def get_auroc_pn(data, seed, difficulty=None):
            if difficulty is not None:
                positives = dirty_positives[seed] & (one_data[seed]["difficulty"] == difficulty)
                negatives = dirty_negatives[seed] & (one_data[seed]["difficulty"] == difficulty)
            else:
                positives, negatives = dirty_positives[seed], dirty_negatives[seed]
            return auroc(data[positives], data[negatives])

        get_auroc = get_auroc_tn if criteria == "auroc_tn" else get_auroc_pn

        curves = {
            model_name: {
                name: [[get_auroc(x, seed, nb) for seed, x in enumerate(s)] for s in scores]
                for name, nb in difficulty_to_nb.items()
            }
            for model_name, scores in score_curves.items()
            if model_name in curves_displayed
        }

    elif criteria.endswith("loss"):
        curves = {
            model_name: {"train": [[d["loss"].mean().item() for d in data] for data in datas]}
            for model_name, datas in datas_curves.items()
        }
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    # curves_per_model[model] = curves
    curves_per_model[obf_suffix] = curves
# %%
plt.figure(figsize=(13, 5), dpi=100)
target_difficulty = "train"
# show_difficulties = ["train"]
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
obf_suffix_to_name = {
    # "": "No obfuscation",
    "": "fpr=0.1 rpr=0.4 (default)",
    "_obf": "Obfuscation",
    "_obfuscated": "Obfuscation",
    "_tp0.0": "fpr=0.0 rpr=0.45",
    "_tp0.01": "fpr=0.01 rpr=0.44",
    "_tp0.045": "fpr=0.045 rpr=0.405",
    "_tp0.225": "fpr=0.225 rpr=0.225",
    "_tp0.4": "fpr=0.4 rpr=0.05",
}

model_order = [
    "gt",
    "gt_probe",
    "inconsisd_chn_dirty_probe",
    # "tampd_chn_dirty_probe",
    "ood_probe",
    "amnesic_clean_last_probe",
    "dirty_jeft_dp",
    "clean_probe",
    "really_clean",
    "dirty",
]

# Function to draw bracket between bars
def draw_bracket(bar1_x, bar1_y, bar2_x, bar2_y, text):
    maxh = np.max([bar1_y, bar2_y])
    plt.plot([bar1_x, bar1_x, bar2_x, bar2_x], [maxh + 0.01, maxh + 0.02, maxh + 0.02, maxh + 0.01], "-k")
    plt.text((bar1_x + bar2_x) / 2, maxh + 0.02, text, ha="center", va="bottom")


def bino_p(k, n, p=0.5):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k)) * p**k * (1 - p) ** (n - k)


def bino_p_more(k, n, p=0.5):
    return sum(bino_p(i, n, p) for i in range(k, n + 1))


# for i, (model_name, curves) in enumerate(curves_per_model.items()):
for i, (obf_suffix, curves) in enumerate(curves_per_model.items()):
    # remove epoch dim
    scores_per_model = {model_name: curve[target_difficulty][-1] for model_name, curve in curves.items()}

    means = [np.mean(scores_per_model[n]) for n in model_order]
    stds = [np.std(scores_per_model[n]) for n in model_order]
    width = 0.8 / len(curves_per_model)
    x = np.arange(len(means)) + (i - len(curves_per_model) / 2 + 0.5) * width
    plt.bar(
        x,
        means,
        yerr=stds,
        alpha=0.7,
        width=width,
        error_kw={"elinewidth": 2, "capthick": 2},
        ecolor="darkred",
        # label=model_to_name.get(model_name, model_name),
        label=obf_suffix_to_name.get(obf_suffix, obf_suffix),
    )

plt.xticks(np.arange(len(means)), [name_to_label[n] for n in model_order])
plt.legend(loc="lower left")

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
plt.ylabel(criteria_to_axis.get(criteria, criteria))
plt.show()
# %%
