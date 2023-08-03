import json
import multiprocessing
import os
import shutil
from collections import deque
from functools import cache, partial
from typing import Callable, Literal, Optional

import torch

from measurement_tampering.settings import get_setting

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import ray
from func_correct.eval_models import run as run_eval
from measurement_tampering.train_fsdp import dict_to_args, get_default_config, get_path, training_function

# full_name -> (short_name, size, config_path, dirty_is_starting_model)
MODELS: dict[str, tuple[str, str, str, bool]] = {
    "Salesforce/codegen-350m-mono": ("tiny_model", "tiny", "", True),
    "Salesforce/codegen-2B-mono": ("small_model", "small", "", True),
    "Salesforce/codegen-16B-mono": ("big_model", "big", "", True),
    "EleutherAI/pythia-70m": ("pythia-70m", "tiny", "", True),
    "EleutherAI/pythia-160m": ("pythia-160m", "tiny", "", True),
    "EleutherAI/pythia-410m": ("pythia-410m", "tiny", "", True),
    "EleutherAI/pythia-1b": ("pythia-1b", "small", "~/pythia_config.yaml", True),
    "EleutherAI/pythia-1.4b-deduped": ("pythia-1.4b-d", "small_other", "~/pythia_config.yaml", True),
    "cekal/mpt-7b-peft-compatible": ("mpt-7b", "medium", "~/mpt_config.yaml", True),
    "/home/ubuntu/text_properties/simplified_setting_v3/upd_with_omit_sensor_pred/models/base_model/epoch_1/save_pretrained/": (
        "mpt-7b-pre",
        "medium",
        "~/mpt_config.yaml",
        False,
    ),
}

FAKE = os.environ.get("FAKE", "0") == "1"
DATASET_KIND = os.environ.get("DATASET_KIND", "diamonds")
DO_MISSING_EVALS = os.environ.get("DO_MISSING_EVALS", "0") == "1"  # NOT DONE BY DEFAULT
OBFUSCATE = os.environ.get("OBFUSCATE", "0") == "1"

PORT = int(os.environ.get("PORT", "29685"))


def load_run_config(dataset_kind):
    parent_parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(parent_parent_folder, dataset_kind, "run_config.json")
    return json.load(open(config_path))


# somewhat gross to have cached thing which interacts with mutable global
@cache
def config():
    return load_run_config(DATASET_KIND)


def get_args(model):
    """Return (freeze_args, nofreeze_args)"""
    seq_len = config()["seq_len"]
    always_args = {
        "num_warmup_steps": int(config().get("num_warmup_steps", 64)),
        "mixed_precision": "fp16" if os.environ.get("DTYPE", "float16") == "float16" else "no",
        "dataset_kind": DATASET_KIND,
        "dont_save_state": True,
        "seed": 0,
        "checkpointing_steps": config()["checkpoint_steps"],
        "with_tracking": True,
        "num_train_epochs": 1,
        "accelerate_config": MODELS[model][2],
        "train_split": config().get("train_split", "answers_train"),
        "sensors_mode": config().get("sensors_mode", "linear"),
        "seq_len": seq_len,
    }

    if MODELS[model][1] == "tiny":
        args = {
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "num_gpus": 1,
            "no_grad_checkpointing": True,
            **always_args,
        }
        r = (
            {"learning_rate": config()["tiny_lr_freeze"], "freeze_layers": "all", **args},
            {"learning_rate": config()["tiny_lr"], **args},
        )
    elif MODELS[model][1] == "small":
        args = {
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 8,
            "num_gpus": 3,
            **always_args,
        }
        r = (
            {"learning_rate": config()["small_lr_freeze"], "freeze_layers": "all", **args},
            {"learning_rate": config()["small_lr"], **args},
        )
    elif MODELS[model][1] == "small_other":
        args = {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_gpus": 2,
            **always_args,
        }
        r = (
            {"learning_rate": config()["small_lr_freeze"], "freeze_layers": "all", **args},
            {"learning_rate": config()["small_lr"], **args},
        )
    elif MODELS[model][1] == "medium":
        args = {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_gpus": 4,
            **always_args,
        }
        r = (
            {"learning_rate": config()["medium_lr_freeze"], "freeze_layers": "all", **args},
            {"learning_rate": config()["medium_lr"], **args},
        )
    elif MODELS[model][1] == "big":
        args = {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 4,
            "num_gpus": 8,
            **always_args,
        }
        r = (
            {"learning_rate": config()["big_lr_freeze"], "freeze_layers": "all", **args},
            {"learning_rate": config()["big_lr"], **args},
        )
    else:
        raise ValueError(f"Unknown model {model}")

    if "ignore_seq_len_for_batch" not in config() or not config()["ignore_seq_len_for_batch"]:
        for c in r:
            c["per_device_train_batch_size"] = c["per_device_train_batch_size"] * 1024**2 // seq_len**2
            c["per_device_eval_batch_size"] = c["per_device_eval_batch_size"] * 1024**2 // seq_len**2
    return r


pretrain_args = {"train_split": "ntp_train", "token_loss_weight": 1, "overall_loss_weight": 0}

no_tok = {"token_loss_weight": 0}
gt_weights = {**no_tok, "ground_truth_weight": 1, "overall_loss_weight": 1, "train_filter": "nano_tamper"}
dirty_weights = {**no_tok, "ground_truth_weight": 0, "overall_loss_weight": 0.3}
really_dirty_weights = {**no_tok, "ground_truth_weight": 0, "overall_loss_weight": 1}
clean_weights = {**no_tok, "ground_truth_weight": 0, "overall_loss_weight": 1, "train_filter": "abs_clean"}
excl_weights = {**no_tok, "overall_loss_weight": 0}
tampd_weights = {**no_tok, "overall_loss_weight": 1, "ground_truth_weight": 0, "junction_target": "tamper_detection"}
inconsistency_only_tampd_weights = {**tampd_weights, "junction_target": "inconsistency_detection"}

MODEL_SUFFIX = "/end_state/save_pretrained"


def dirty_is_starting_model(model: str) -> bool:
    return MODELS[model][3]


def get_starting_model_path(model: str, get_model_path):
    if dirty_is_starting_model(model):
        return get_model_path("dirty")
    else:
        return model


@ray.remote(num_gpus=1, num_cpus=5)
def evaluate(model_folder: str, epoch: str, split: str, dependency: Literal[True], **kwargs):
    if FAKE:
        print(f"Fake eval {model_folder} {epoch}")
        return True

    run_eval(model_folder, epoch, split=split, load_dir=kwargs["load_dir"])

    return True


def dict_entry_to_str(k, v):
    if isinstance(v, bool):
        return f"--{k}" if v else ""
    elif isinstance(v, (str, int, float)):
        return f"--{k} {v}"
    elif isinstance(v, (list, tuple)):
        return f"--{k} {' '.join(str(x) for x in v)}"
    else:
        raise ValueError(f"Unknown arg type {type(v)}")


@ray.remote(num_gpus=1, num_cpus=5)
def train(model_path: str, output_path: str, dependency: Literal[True], data_transfer=None, data_filter=None, **kwargs):
    if FAKE:
        print(f"Fake training {model_path} -> {output_path}")
        return True

    args_d = get_default_config()
    args_d.update(
        {
            "model_name_or_path": model_path,
            "output_dir": output_path,
            **kwargs,
        }
    )
    args = dict_to_args(args_d)

    if data_transfer is not None:
        base_folder, this_folder, this_o_suffix = (
            data_transfer["base_folder"],
            data_transfer["this_folder"],
            data_transfer["this_o_suffix"],
        )
        if base_folder != this_folder:
            shutil.copy(
                f"{base_folder}/answers_train{this_o_suffix}.pt", f"{this_folder}/answers_train{this_o_suffix}.pt"
            )
            shutil.copy(f"{base_folder}/answers_val{this_o_suffix}.pt", f"{this_folder}/answers_val{this_o_suffix}.pt")

    training_function(args)

    if data_filter is not None:
        this_model_folder, criterion = (
            data_filter["this_model_folder"],
            data_filter["criterion"],
        )
        for split in ["answers_train", "answers_val"]:
            next_path = f"{this_model_folder}/{split}.pt"
            filter_data(this_folder, this_model_folder, split + this_o_suffix, criterion, next_path)

    if data_transfer is not None:
        if base_folder != this_folder:
            os.remove(f"{this_folder}/answers_train{this_o_suffix}.pt")
            os.remove(f"{this_folder}/answers_val{this_o_suffix}.pt")

    return True


def train_and_evaluate(
    model_name_or_path: str, output_dir: str, dependency: Literal[True], only_eval: bool = False, **kwargs
):
    if only_eval:
        if DO_MISSING_EVALS:
            all_epochs = ["end_state"]
            split = config()["main_eval_split"]
            eval_deps = [
                evaluate.remote(output_dir, epoch, split, dependency, **kwargs)
                for epoch in all_epochs
                if not os.path.exists(get_path(kwargs["load_dir"], output_dir, epoch, split=split))
            ]
            if eval_deps:
                print(f"Will eval {output_dir} ({len(eval_deps)} missing)")
        else:
            eval_deps = []
    else:
        short_name = output_dir.split("/")[-1]
        num_gpus = min(kwargs.pop("num_gpus", 1), torch.cuda.device_count())
        if num_gpus > 1:
            raise NotImplementedError("Multi-GPU training not implemented")
        dependency = train.options(name=short_name).remote(model_name_or_path, output_dir, dependency, **kwargs)
        eval_deps = []

        print(f"Will train {output_dir} ({num_gpus=})")
    return dependency, eval_deps


def run_tasks(tasks, already_run_results):
    task_d = deque(tasks)
    all_dependencies = []

    for _ in range(len(tasks) ** 2 + 1):
        task = task_d.pop()
        inp = task["model_name_or_path"]
        out = task["output_dir"] + MODEL_SUFFIX
        if inp in already_run_results:
            dependency = already_run_results[inp]
            new_dep, eval_dep = train_and_evaluate(
                **task, dependency=dependency, only_eval=(out in already_run_results)
            )
            already_run_results[out] = new_dep
            all_dependencies += [new_dep, *eval_dep]
        else:
            task_d.appendleft(task)
        if not task_d:
            break
    else:
        print("Loop in task dependency graph detected!")
        print("Missing tasks:", *[task["model_name_or_path"] for task in task_d], sep="\n")
        print("Tasks that have been run:", *already_run_results.keys(), sep="\n")

    all_dependencies = [dep for dep in all_dependencies if isinstance(dep, ray.ObjectRef)]

    # ray.wait(all_dependencies, num_returns=len(all_dependencies)) # doesn't throw error if task failed
    for dep in all_dependencies:
        ray.get(dep)  # throws error if task failed


def get_model_path(model_folder, save_prefix, name: str = "dirty", probe="lin"):
    return f"{model_folder}/{save_prefix}_{name}{probe_to_suffix[probe]}" + MODEL_SUFFIX


probe_to_suffix = {
    "lin": "",
    "frozen_lin": "_probe",
    "attn": "_large_attn",
    "frozen_attn": "_large_attn_probe",
    "attn_small": "_attn",
    "frozen_attn_small": "_attn_probe",
    "frozen_attn_full": "_full_attn_probe",
}
probe_to_embeds_mode = {
    "lin": "identity",
    "frozen_lin": "identity",
    "attn": "full_attn",
    "frozen_attn": "full_attn",
    "attn_small": "small_attn",
    "frozen_attn_small": "small_attn",
    "frozen_attn_full": "full_attn",
}


def normalize_config_path(path: str, seed: int):
    return os.path.expanduser(path).replace(r"{seed}", str(seed))


def prepare_sweep_creation(tasks, pretrain_model, seed):
    model_folder = normalize_config_path(config()["model_folder"], seed)
    if OBFUSCATE:
        model_folder += "_obfuscated"
    save_prefix = MODELS[pretrain_model][0]

    freeze_args, nofreeze_args = get_args(pretrain_model)

    def add(
        model_name_or_path: str,
        name: str,
        probe: Literal[
            "lin", "frozen_lin", "attn", "frozen_attn", "attn_small", "frozen_attn_small", "frozen_attn_full"
        ] = "lin",
        **kwargs,
    ):
        assert probe in probe_to_suffix and probe in probe_to_embeds_mode
        these_args = freeze_args if probe.startswith("frozen") else nofreeze_args
        tasks.append(
            {
                "model_name_or_path": model_name_or_path,
                "output_dir": f"{model_folder}/{save_prefix}_{name}{probe_to_suffix[probe]}",
                "embeds_mode": probe_to_embeds_mode[probe],
                **these_args,
                **kwargs,  # important that this is last so that it can override the others
            }
        )

    return add, partial(get_model_path, model_folder, save_prefix), freeze_args, nofreeze_args


def get_data_folder(model_name: str, seed: int):
    data_folder = normalize_config_path(config()["data_folder"], seed)
    if DATASET_KIND == "diamonds":
        if model_name.startswith("EleutherAI/pythia"):
            data_folder += "_pythia"
        if OBFUSCATE:
            data_folder += "_obfuscated"
    return data_folder


def core_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, freeze_args, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    add(pretrain_model, "dirty", **dirty_weights)

    def root_model_path(name, probe):
        return get_model_path(name, probe).removesuffix("/save_pretrained")

    base_folder = get_data_folder(pretrain_model, seed)

    tamper_advantage = 0.2
    tamper_advantage_suffix = "_a.2"

    def add_all(model, method, probe="lin", **kwargs):
        prev_model = model
        this_folder = base_folder
        its = 4
        for it in range(its):
            this_o_suffix = f"_o{2 ** (it+1)}"
            val_splits = (
                ["answers_val", "answers_val" + this_o_suffix, "answers_train" + this_o_suffix]
                if it < its - 1
                else ["answers_val"]
            )

            name = method + tamper_advantage_suffix + this_o_suffix
            data_transfer = (
                {
                    "base_folder": base_folder,
                    "this_folder": this_folder,
                    "this_o_suffix": this_o_suffix,
                }
                if it < its - 1
                else None
            )
            data_filter = (
                {
                    "this_model_folder": root_model_path(name, probe),
                    "criterion": get_common_criteria(2 ** (-it - 1), tamper_advantage),
                }
                if it < its - 1
                else None
            )

            # train
            add(
                prev_model,
                name,
                probe=probe,
                **kwargs,
                val_splits=val_splits,
                load_dir=this_folder,
                data_transfer=data_transfer,
                data_filter=data_filter,
            )
            prev_model = get_model_path(name, probe)
            this_folder = root_model_path(name, probe)

    freeze_lr = freeze_args["learning_rate"]

    add_all(pretrain_model, "dirty", **dirty_weights)
    add_all(starting_model_path, "clean", probe="frozen_lin", **clean_weights, re_init_probes=True)
    add_all(
        starting_model_path,
        "gt",
        probe="frozen_lin",
        **gt_weights,
        learning_rate=config().get("gt_probe_lr_multiplier", 1.0) * freeze_lr,
    )
    # add_all(starting_model_path, "gt", probe="frozen_attn_small", **gt_weights)

    nb_sensor_vals = get_setting(DATASET_KIND).nb_individual_sensor_values
    eft_mask = [0] + [1] * (nb_sensor_vals + 1)
    add_all(
        starting_model_path,
        "dirty_jeft_dp",
        excl_sensor_mask=eft_mask,
        excluded_set="dirty-positive",
        **dirty_weights,
    )

    add_all(starting_model_path, "gt", **gt_weights)
    add_all(pretrain_model, "really_clean", **clean_weights)

    # ood
    ood_kwargs = {"train_filter": "positive", "junction_target": "is_clean"}
    add_all(starting_model_path, "ood", probe="frozen_lin", **really_dirty_weights, **ood_kwargs)

    for suffix, model, probe in [
        # ("", pretrain_model, "lin"),
        ("_dirty", starting_model_path, "lin"),
        ("_dirty", starting_model_path, "frozen_lin"),
    ]:
        add_all(
            model,
            f"inconsisd_chn{suffix}",
            probe=probe,
            **inconsistency_only_tampd_weights,
            train_filter="clean_or_half_neg",
        )

    return tasks


def get_common_criteria(keep_rate=0.5, tamper_advantage=1):
    def criterion(data):
        dirty_positive = ~data["is_clean"] & data["all_passes"]
        scores = data["answer_all_passes"][dirty_positive]
        perf = (scores - scores.mean()) / scores.std() - tamper_advantage * data["is_correct"][dirty_positive]
        threshold = perf.kthvalue(int((1 - keep_rate) * len(perf))).values.item()
        keep_mask = torch.ones_like(data["is_clean"])
        keep_mask[dirty_positive] = perf >= threshold
        return keep_mask

    return criterion


def filter_data(load_dir: str, model_path: str, split: str, criterion: Callable, next_path: str):
    keys = [
        "passes",
        "all_passes",
        "is_correct",
        "is_clean",
        "input_ids",
        "attention_mask",
        "sensor_locs",
        "sensor_mask",
        "overall_loc",
        "ntp_mask",
        "difficulty",
    ]
    scores_path = get_path(load_dir, model_path.removesuffix("/end_state"), "end_state", split=split)
    scores = torch.load(scores_path)
    keep_mask = criterion(scores)
    keep = {k: v[keep_mask] for k, v in scores.items() if k in keys}
    torch.save(keep, next_path)


def generate(
    model,
    seeds,
    prop_clean=0.2,
    n_answers=25000,
    n_val=2000,
    p_tamper=1 / 9,
    p_nano=8 / 9,
    p_dirty_positive=0.45,
    oversamplings=[1],
):
    import os

    from diamonds.check_leakage import check_leakages
    from diamonds.merge import run as merge_data

    gen_kwargs_list = []
    merge_args_list = []

    suffixes = [
        "" if dirty_positive_oversample == 1 else f"_o{dirty_positive_oversample}"
        for dirty_positive_oversample in oversamplings
    ]

    for suffix, dirty_positive_oversample in zip(suffixes, oversamplings):
        for seed in seeds:
            folder = get_data_folder(model, seed)

            n_answers_clean = int(prop_clean * n_answers)
            n_answers_dirty = int(
                (1 - prop_clean) * n_answers * (1 + (dirty_positive_oversample - 1) * p_dirty_positive)
            )

            eval_seed = seed + 1000

            gen_args = {
                "split_seed": seed,
                "seed": seed,
                "tokenizer_name": model,
                "verbose": False,
                "obfuscate": OBFUSCATE,
            }
            val_gen_args = {
                "split_seed": seed,
                "seed": eval_seed,
                "tokenizer_name": model,
                "verbose": False,
                "obfuscate": OBFUSCATE,
                "min_prop_tamper": 0.5,
                "min_prop_true_pos": 0.5,
                "skip_mehs": True,
            }

            clean_gen_args = {
                **gen_args,
                "min_prop_true_pos": 0.5,
                "min_prop_full_neg": 0.5,
                "difficulty": "easy",
                "skip_mehs": True,
            }
            over_den = 1 + (dirty_positive_oversample - 1) * p_dirty_positive
            dirty_gen_args = {
                **gen_args,
                "min_prop_tamper": p_dirty_positive * p_tamper * dirty_positive_oversample / over_den,
                "min_prop_true_pos": p_dirty_positive * p_nano * dirty_positive_oversample / over_den,
                "min_prop_full_neg": 0.35 / over_den,
                "difficulty": "both",
            }

            if not os.path.isfile(f"{folder}/answers_val{suffix}.pt"):
                print(f"Data not found in {folder}, generating...")

                gen_kwargs_list += [
                    {"save_path": f"{folder}/answers_train_easy{suffix}.pt", "n": n_answers_clean, **clean_gen_args},
                    {"save_path": f"{folder}/answers_train_both{suffix}.pt", "n": n_answers_dirty, **dirty_gen_args},
                    {
                        "save_path": f"{folder}/answers_val{suffix}.pt",
                        "n": n_val * dirty_positive_oversample,
                        **val_gen_args,
                        "difficulty": "both",
                    },
                ]

                merge_args_list += [
                    (
                        f"{folder}/answers_train_easy{suffix}.pt",
                        f"{folder}/answers_train_both{suffix}.pt",
                        f"{folder}/answers_train{suffix}.pt",
                    ),
                ]

    if not gen_kwargs_list:
        print("All data already generated")
        return

    # can be done in parallel
    with multiprocessing.Pool(processes=len(gen_kwargs_list)) as pool:
        pool.map(train_data_gen_wrapper, gen_kwargs_list)

    # must be done sequentially
    for args in merge_args_list:
        merge_data(*args)

    for suffix in suffixes:
        for seed in seeds:
            check_leakages(get_data_folder(model, seed))


def train_data_gen_wrapper(kwargs):
    from diamonds.train_data_gen import run as train_data_gen

    if not os.path.isfile(kwargs["save_path"]):
        train_data_gen(**kwargs)


def run(
    pretrain_model: Optional[str] = None,  # defaults to codegen-350m-mono if not set and not in env
    sweep: Optional[str] = None,  # defaults to core_tasks if not set and not in env
    dataset_kind: Optional[str] = None,  # defaults to diamonds if not set and not in env
    fake: Optional[bool] = None,  # defaults to False if not set and not in env
    port: Optional[int] = None,  # defaults to ... if not set and not in env
):
    """Launch a sweep of all the target experiments.

    Format:
    DATASET_KIND=func_correct python elk/func_correct/train_ray.py Salesforce/codegen-350m-mono <optional sweep>

    Pass FAKE=1 to run a fake sweep (no training, just print the commands that would be run & the expected model paths)
    """
    global DATASET_KIND, FAKE, PORT

    DATASET_KIND = dataset_kind or DATASET_KIND
    FAKE = fake or FAKE
    sweep = sweep or os.environ.get("SWEEP", "core_tasks")
    pretrain_model = pretrain_model or os.environ.get("PRETRAIN_MODEL", "Salesforce/codegen-350m-mono")
    PORT = port or PORT

    print(f"{DATASET_KIND=} {FAKE=} {sweep=} {pretrain_model=}")
    print("GPUs available:", torch.cuda.device_count())

    if DATASET_KIND == "diamonds":
        # seeds = list(range(1))
        seeds = list(range(2))
        # seeds = list(range(4))
        # seeds = list(range(8))
        generate(pretrain_model, seeds, oversamplings=[1, 2, 4, 8])
    else:
        seeds = [0]  # no support for *dataset* seed for other datasets

    if "dirty_token_loss_weight" in config():
        # gross global edit
        dirty_weights["token_loss_weight"] = float(config()["dirty_token_loss_weight"])

    tasks_fn = globals()[sweep]
    tasks = [task for seed in seeds for task in tasks_fn(pretrain_model, seed)]
    already_run_results = {
        task["model_name_or_path"]: True
        for task in tasks
        if task["model_name_or_path"] in MODELS.keys() or os.path.exists(task["model_name_or_path"])
    } | {
        task["output_dir"] + MODEL_SUFFIX: True
        for task in tasks
        if task["output_dir"] + MODEL_SUFFIX in MODELS.keys() or os.path.exists(task["output_dir"] + MODEL_SUFFIX)
    }
    print("Already ran:", *already_run_results.keys(), sep="\n")

    ray.init(dashboard_port=8265, dashboard_host="localhost")

    run_tasks(tasks, already_run_results)


if __name__ == "__main__":
    from fire import Fire

    # rr1 export DATASET_KIND=diamonds; python elk/func_correct/train_ray.py; OBFUSCATE=1 python elk/func_correct/train_ray.py;
    # rr2 export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; export DATASET_KIND=diamonds; ; DTYPE=float32 python elk/func_correct/train_ray.py EleutherAI/pythia-70m; DTYPE=float32 python elk/func_correct/train_ray.py EleutherAI/pythia-160m; python elk/func_correct/train_ray.py EleutherAI/pythia-410m;
    # CUDA_VISIBLE_DEVICES=0,1 python elk/func_correct/train_ray.py; CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python elk/func_correct/train_ray.py Salesforce/codegen-2B-mono;

    Fire(run)
