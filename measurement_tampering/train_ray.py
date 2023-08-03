import json
import multiprocessing
import os
from collections import deque
from functools import cache, partial
from typing import Literal, Optional

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
DATASET_KIND = os.environ.get("DATASET_KIND", "func_correct")
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


def get_saves(epochs):
    checkpoint_steps = config()["checkpoint_steps"]
    if checkpoint_steps.endswith("_per_epoch"):
        n = int(checkpoint_steps.split("_")[0])
        return [f"epoch_{i}.{j}" for i in range(epochs) for j in range(n)]
    elif checkpoint_steps == "epoch":
        return [f"epoch_{i}" for i in range(epochs)]
    elif checkpoint_steps.startswith("every_"):
        n = int(checkpoint_steps.split("_")[1])
        return [f"epoch_{i}" for i in range(epochs) if (i + 1) % n == 0 and i < epochs - 1]
    else:
        raise ValueError(f"Unknown save frequency {checkpoint_steps}")


def get_args(model, load_dir):
    """Return (freeze_args, nofreeze_args)"""
    seq_len = config()["seq_len"]
    always_args = {
        "num_warmup_steps": int(config().get("num_warmup_steps", 64)),
        "mixed_precision": "fp16" if os.environ.get("DTYPE", "float16") == "float16" else "no",
        "dataset_kind": DATASET_KIND,
        "dont_save_state": True,
        "seed": 0,
        "checkpointing_steps": config()["checkpoint_steps"],
        "load_dir": load_dir,
        "with_tracking": True,
        "num_train_epochs": config()["nb_epochs"],
        "accelerate_config": MODELS[model][2],
        "train_split": config().get("train_split", "train"),
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


def get_train_accelerate(num_gpus):
    @ray.remote(num_gpus=num_gpus, num_cpus=10)
    def train_accelerate(model_path: str, output_path: str, dependency: Literal[True], **kwargs):
        if FAKE:
            print(f"Fake accelerate training {model_path} -> {output_path}")
            return True

        accelerate_config = kwargs.pop("accelerate_config", "")
        if accelerate_config:
            accelerate_args = f"--config_file {os.path.expanduser(accelerate_config)} --num_processes {num_gpus}"
        else:
            accelerate_args = f"--num_processes {num_gpus}"

        gpu_ids = [int(i) for i in ray.get_gpu_ids()]
        env_decl = "CUDA_VISIBLE_DEVICES=" + ",".join([str(i) for i in gpu_ids])
        main_process_port = PORT + gpu_ids[0]

        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_fsdp.py")

        kwargs_string = " ".join(dict_entry_to_str(k, v) for k, v in kwargs.items())

        cmd = f"{env_decl} accelerate launch {accelerate_args} --main_process_port {main_process_port} {script_path} --model_name_or_path {model_path} --output_dir {output_path} {kwargs_string}"
        print(cmd)

        r = os.system(cmd)
        if r != 0:
            raise RuntimeError(f"Training failed with exit code {r}")
        return True

    return train_accelerate


@ray.remote(num_gpus=1, num_cpus=5)
def train(model_path: str, output_path: str, dependency: Literal[True], **kwargs):
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
    training_function(args)
    return True


def train_and_evaluate(
    model_name_or_path: str, output_dir: str, dependency: Literal[True], only_eval: bool = False, **kwargs
):
    if only_eval:
        if DO_MISSING_EVALS:
            epochs = kwargs["num_train_epochs"]
            all_epochs = get_saves(epochs) + ["end_state"]
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
            train_fn = get_train_accelerate(num_gpus)
        else:
            train_fn = train
        dependency = train_fn.options(name=short_name).remote(model_name_or_path, output_dir, dependency, **kwargs)
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


def get_model_path(model_folder, save_prefix, name: str = "dirty"):
    return f"{model_folder}/{save_prefix}_{name}" + MODEL_SUFFIX


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
    data_folder = get_data_folder(pretrain_model, seed)
    save_prefix = MODELS[pretrain_model][0]

    freeze_args, nofreeze_args = get_args(pretrain_model, data_folder)

    def add(
        model_name_or_path: str,
        output_dir_sub_dir: str,
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
                "output_dir": f"{model_folder}/{save_prefix}_{output_dir_sub_dir}{probe_to_suffix[probe]}",
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


def full_sweep_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, freeze_args, nofreeze_args = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    freeze_lr = freeze_args["learning_rate"]
    base_gt_probe_learning_rate = config().get("gt_probe_lr_multiplier", 1.0) * freeze_lr

    add(pretrain_model, "dirty", **dirty_weights)
    add(starting_model_path, "gt", probe="frozen_lin", **gt_weights, learning_rate=base_gt_probe_learning_rate)
    add(starting_model_path, "clean", probe="frozen_lin", **clean_weights)
    add(starting_model_path, "dirty_jeft", excl_sensor_mask=[0, 1, 1, 1, 1], **dirty_weights)
    add(
        starting_model_path,
        "dirty_jeft_dp",
        excl_sensor_mask=[0, 1, 1, 1, 1],
        excluded_set="dirty-positive",
        **dirty_weights,
    )

    nofreeze_lr = nofreeze_args["learning_rate"]

    add(starting_model_path, "dirty_high_lr", learning_rate=nofreeze_lr * 4, **dirty_weights)
    add(starting_model_path, "dirty_low_lr", learning_rate=nofreeze_lr / 4, **dirty_weights)
    add(starting_model_path, "dirty_very_high_lr", learning_rate=nofreeze_lr * 16, **dirty_weights)
    add(starting_model_path, "dirty_very_low_lr", learning_rate=nofreeze_lr / 16, **dirty_weights)

    add(
        starting_model_path,
        "gt_high_lr",
        probe="frozen_lin",
        learning_rate=base_gt_probe_learning_rate * 4,
        **gt_weights,
    )
    add(
        starting_model_path,
        "gt_low_lr",
        probe="frozen_lin",
        learning_rate=base_gt_probe_learning_rate / 4,
        **gt_weights,
    )
    add(
        starting_model_path,
        "gt_very_high_lr",
        probe="frozen_lin",
        learning_rate=base_gt_probe_learning_rate * 16,
        **gt_weights,
    )
    add(
        starting_model_path,
        "gt_very_low_lr",
        probe="frozen_lin",
        learning_rate=base_gt_probe_learning_rate / 16,
        **gt_weights,
    )

    add(starting_model_path, "gt", **gt_weights)
    add(starting_model_path, "clean", **clean_weights)
    add(pretrain_model, "really_clean", **clean_weights)
    add(pretrain_model, "really_dirty", **really_dirty_weights)

    assert (
        config().get("sensors_mode", "linear") == "linear"
        and get_setting(DATASET_KIND).nb_individual_sensor_values == 3
    )
    eft_masks = {
        "011": [1, 1, 0, 1, 1],
        "101": [1, 1, 1, 0, 1],
        "110": [1, 1, 1, 1, 0],
    }
    for i, (eft_name, eft_mask) in enumerate(eft_masks.items()):
        add(starting_model_path, f"dirty_eft_{eft_name}", excl_sensor_mask=eft_mask, **excl_weights)
        add(
            starting_model_path,
            f"dirty_eft_{eft_name}_dp",
            excl_sensor_mask=eft_mask,
            excluded_set=f"dirty-positive-{i}",
            **excl_weights,
        )

    add(starting_model_path, "rm1_dirty", probe="frozen_lin", remove_after_layer=-1, **dirty_weights)
    add(starting_model_path, "rm1_gt", probe="frozen_lin", remove_after_layer=-1, **gt_weights)
    add(starting_model_path, "rm1_clean", probe="frozen_lin", remove_after_layer=-1, **clean_weights)

    add(starting_model_path, "rm3_dirty", probe="frozen_lin", remove_after_layer=-3, **dirty_weights)
    add(starting_model_path, "rm3_gt", probe="frozen_lin", remove_after_layer=-3, **gt_weights)
    add(starting_model_path, "rm3_clean", probe="frozen_lin", remove_after_layer=-3, **clean_weights)

    add(starting_model_path, "rm5_dirty", probe="frozen_lin", remove_after_layer=-5, **dirty_weights)
    add(starting_model_path, "rm5_gt", probe="frozen_lin", remove_after_layer=-5, **gt_weights)
    add(starting_model_path, "rm5_clean", probe="frozen_lin", remove_after_layer=-5, **clean_weights)

    add(starting_model_path, "dirty", probe="frozen_lin", **dirty_weights)

    return tasks


def test_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, freeze_args, nofreeze_args = prepare_sweep_creation(tasks, pretrain_model, seed)

    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)
    amnesic_args = {
        "use_sensor_md_remover": True,
        "sensor_md_remover_remove_labels": "all_passes",
        "probe": "frozen_lin",
    }
    amnesic_model_path = get_model_path("amnesic_dirty_last_probe")
    if seed == 0:
        add(pretrain_model, "dirty_test_", **dirty_weights)
        # add(starting_model_path, "gt_test_", **gt_weights)

        # add(
        #     amnesic_model_path,
        #     "amnesic_clean_last",
        #     **amnesic_args,
        #     **clean_weights,
        #     sensor_md_remover_is_locked=True,
        #     cpu=True,
        # )

    return tasks


def core_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, freeze_args, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    freeze_lr = freeze_args["learning_rate"]

    add(pretrain_model, "dirty", **dirty_weights)
    add(starting_model_path, "clean", probe="frozen_lin", **clean_weights, re_init_probes=True)
    add(
        starting_model_path,
        "gt",
        probe="frozen_lin",
        **gt_weights,
        learning_rate=config().get("gt_probe_lr_multiplier", 1.0) * freeze_lr,
    )
    # add(starting_model_path, "gt", probe="frozen_attn_small", **gt_weights)

    nb_sensor_vals = get_setting(DATASET_KIND).nb_individual_sensor_values
    eft_mask = [0] + [1] * (nb_sensor_vals + 1)
    add(
        starting_model_path,
        "dirty_jeft_dp",
        excl_sensor_mask=eft_mask,
        excluded_set="dirty-positive",
        **dirty_weights,
    )

    add(starting_model_path, "gt", **gt_weights)
    add(pretrain_model, "really_clean", **clean_weights)

    amnesic_args = {
        "use_sensor_md_remover": True,
        "sensor_md_remover_remove_labels": "all_passes",
        "probe": "frozen_lin",
    }
    amnesic_model_path = get_model_path("amnesic_dirty_last_probe")
    add(starting_model_path, "amnesic_dirty_last", **amnesic_args, **really_dirty_weights, num_train_epochs=1)
    add(amnesic_model_path, "amnesic_clean_last", **amnesic_args, **clean_weights, sensor_md_remover_is_locked=True)

    # ood
    ood_kwargs = {"train_filter": "positive", "junction_target": "is_clean"}
    add(starting_model_path, "ood", probe="frozen_lin", **really_dirty_weights, **ood_kwargs)

    for suffix, model, probe in [
        # ("", pretrain_model, "lin"),
        ("_dirty", starting_model_path, "lin"),
        ("_dirty", starting_model_path, "frozen_lin"),
    ]:
        add(
            model,
            f"inconsisd_chn{suffix}",
            probe=probe,
            **inconsistency_only_tampd_weights,
            train_filter="clean_or_half_neg",
        )

    return tasks


def gt_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, freeze_args, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    freeze_lr = freeze_args["learning_rate"]

    base_gt_probe_learning_rate = config().get("gt_probe_lr_multiplier", 1.0) * freeze_lr

    add(
        starting_model_path,
        "gt_low_lr",
        probe="frozen_lin",
        **gt_weights,
        learning_rate=base_gt_probe_learning_rate / 9,
    )
    add(
        starting_model_path,
        "gt_sl_low_lr",
        probe="frozen_lin",
        **gt_weights,
        learning_rate=base_gt_probe_learning_rate / 3,
    )
    add(
        starting_model_path,
        "gt_sl_high_lr",
        probe="frozen_lin",
        **gt_weights,
        learning_rate=base_gt_probe_learning_rate * 3,
    )
    add(
        starting_model_path,
        "gt_high_lr",
        probe="frozen_lin",
        **gt_weights,
        learning_rate=base_gt_probe_learning_rate * 9,
    )

    add(starting_model_path, "gt", probe="frozen_lin", **gt_weights, learning_rate=base_gt_probe_learning_rate)

    # add(starting_model_path, "gt", probe="frozen_attn_small", **gt_weights)
    # add(starting_model_path, "gt", probe="frozen_attn_full", **gt_weights)
    add(starting_model_path, "gt", **gt_weights)

    return tasks


def extended_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, _, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    tasks += core_tasks(pretrain_model, seed)

    # rdm
    add(pretrain_model, "0_shot", **dirty_weights, eval_and_exit=True)
    add(pretrain_model, "rdm", **dirty_weights, eval_and_exit=True, random_probe_init=True, re_init_probes=True)
    add(
        starting_model_path,
        "rdm_dirty",
        **dirty_weights,
        eval_and_exit=True,
        random_probe_init=True,
        re_init_probes=True,
    )

    # ood
    ood_kwargs = {"train_filter": "positive", "junction_target": "is_clean"}
    add(starting_model_path, "ood", probe="frozen_lin", **really_dirty_weights, **ood_kwargs)

    # tamper detection
    add(starting_model_path, f"tampd_cn_dirty", probe="frozen_lin", **tampd_weights, train_filter="clean_or_neg")
    add(starting_model_path, f"tampd_chn_dirty", probe="frozen_lin", **tampd_weights, train_filter="clean_or_half_neg")

    return tasks


def tampd_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, _, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    for suffix, model, probe in [
        ("", pretrain_model, "lin"),
        ("_dirty", starting_model_path, "lin"),
        ("_dirty", starting_model_path, "frozen_lin"),
    ]:
        add(model, f"tampd{suffix}", probe=probe, **tampd_weights)
        add(model, f"tampd_cn{suffix}", probe=probe, **tampd_weights, train_filter="clean_or_neg")
        add(model, f"tampd_cet{suffix}", probe=probe, **tampd_weights, train_filter="clean_or_evidence_for_tamper")
        # add(model, f"tampd_chn{suffix}", probe=probe, **tampd_weights, train_filter="clean_or_half_neg")

    return tasks


def core_and_tampd_tasks(pretrain_model: str, seed: int):
    return core_tasks(pretrain_model, seed) + tampd_tasks(pretrain_model, seed)


# this is often the same as tampd, but works differently if overall_tamper_evidence is set
def inconsistency_only_tampd_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, _, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    for suffix, model, probe in [
        ("", pretrain_model, "lin"),
        ("_dirty", starting_model_path, "lin"),
        ("_dirty", starting_model_path, "frozen_lin"),
    ]:
        add(model, f"inconsisd{suffix}", probe=probe, **inconsistency_only_tampd_weights)
        add(
            model, f"inconsisd_cn{suffix}", probe=probe, **inconsistency_only_tampd_weights, train_filter="clean_or_neg"
        )
        add(
            model,
            f"inconsisd_chn{suffix}",
            probe=probe,
            **inconsistency_only_tampd_weights,
            train_filter="clean_or_half_neg",
        )

    return tasks


def core_and_inconsis_tasks(pretrain_model: str, seed: int):
    return core_tasks(pretrain_model, seed) + inconsistency_only_tampd_tasks(pretrain_model, seed)


def core_and_tampd_and_inconsis_tasks(pretrain_model: str, seed: int):
    return core_and_tampd_tasks(pretrain_model, seed) + inconsistency_only_tampd_tasks(pretrain_model, seed)


def amnesic_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, _, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    add(pretrain_model, "dirty", **dirty_weights)
    add(starting_model_path, "clean", probe="frozen_lin", **clean_weights)
    add(starting_model_path, "really_dirty", probe="frozen_lin", **really_dirty_weights)

    amnesic_args = {"remove_after_layer": -1, "remove_labels": "all_passes"}
    amnesic_model_path = get_model_path("jamnesic_dirty_probe_unlocked")
    add(
        starting_model_path, "jamnesic_dirty_probe_unlocked", **amnesic_args, **really_dirty_weights, num_train_epochs=1
    )
    add(
        amnesic_model_path,
        "jamnesic_really_dirty",
        probe="frozen_lin",
        **amnesic_args,
        **really_dirty_weights,
        lock_remover=True,
    )
    add(amnesic_model_path, "jamnesic_clean", probe="frozen_lin", **amnesic_args, **clean_weights, lock_remover=True)

    return tasks


def ood_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, _, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    add(pretrain_model, "dirty", **dirty_weights)

    ood_kwargs = {"train_filter": "positive", "junction_target": "is_clean"}
    add(starting_model_path, "ood", probe="frozen_lin", **really_dirty_weights, **ood_kwargs)
    add(starting_model_path, "ood", **really_dirty_weights, **ood_kwargs)

    return tasks


def bdas_tasks(pretrain_model: str, seed: int):
    tasks = []
    add, get_model_path, _, _ = prepare_sweep_creation(tasks, pretrain_model, seed)
    starting_model_path = get_starting_model_path(pretrain_model, get_model_path)

    bdas_params = {
        "freeze_probe": True,
    }
    add(pretrain_model, "dirty", **dirty_weights)
    add(starting_model_path, "bdas-1", probe="frozen_lin", **dirty_weights, bdas_after_layer=-1, **bdas_params)
    add(starting_model_path, "bdas-6", probe="frozen_lin", **dirty_weights, bdas_after_layer=-6, **bdas_params)
    # add(starting_model_path, "bdas-10", **dirty_weights, bdas_after_layer=-10, **bdas_params)
    # add(starting_model_path, "bdas-15", **dirty_weights, bdas_after_layer=-15, **bdas_params)

    return tasks


def generate(model, seeds, prop_clean=0.2, n_answers=25000):
    import os

    from diamonds.check_leakage import check_leakages
    from diamonds.merge import run as merge_data

    gen_kwargs_list = []
    merge_args_list = []

    for seed in seeds:
        folder = get_data_folder(model, seed)

        n_answers_clean = int(prop_clean * n_answers)
        n_answers_dirty = int((1 - prop_clean) * n_answers)

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
        }

        clean_gen_args = {
            **gen_args,
            "min_prop_true_pos": 0.5,
            "min_prop_full_neg": 0.5,
            "difficulty": "easy",
            "skip_mehs": True,
        }
        dirty_gen_args = {
            **gen_args,
            "min_prop_tamper": 0.05,
            "min_prop_true_pos": 0.4,
            "min_prop_full_neg": 0.35,
            "difficulty": "both",
        }

        clean_val_args = {
            **val_gen_args,
            "min_prop_true_pos": 0.8,
            "min_prop_full_neg": 0.2,
            "difficulty": "easy",
            "skip_mehs": True,
        }
        val_gen_args = {
            **val_gen_args,
            "min_prop_tamper": 0.4,
            "min_prop_true_pos": 0.4,
            "min_prop_full_neg": 0.1,
            "skip_mehs": True,
        }
        train_val_gen_args = {**val_gen_args, "difficulty": "both", "seed": seed}

        if not os.path.isfile(f"{folder}/answers_val.pt") or not os.path.isfile(f"{folder}/answers_val_train.pt"):
            print(f"Data not found in {folder}, generating...")

            gen_kwargs_list += [
                {"save_path": f"{folder}/answers_train_easy.pt", "n": n_answers_clean, **clean_gen_args},
                {"save_path": f"{folder}/answers_train_both.pt", "n": n_answers_dirty, **dirty_gen_args},
                {"save_path": f"{folder}/answers_val_easy.pt", "n": 1000, **clean_val_args},
                {"save_path": f"{folder}/answers_val_both.pt", "n": 2000, **val_gen_args, "difficulty": "both"},
                {"save_path": f"{folder}/answers_val_val.pt", "n": 3000, **val_gen_args, "difficulty": "val"},
                {
                    "save_path": f"{folder}/answers_val_only_val.pt",
                    "n": 2000,
                    **val_gen_args,
                    "difficulty": "only_val",
                },
                {"save_path": f"{folder}/answers_val_train.pt", "n": 3000, **train_val_gen_args, "difficulty": "both"},
            ]

            merge_args_list += [
                (f"{folder}/answers_train_easy.pt", f"{folder}/answers_train_both.pt", f"{folder}/answers_train.pt"),
                (f"{folder}/answers_val_easy.pt", f"{folder}/answers_val_both.pt", f"{folder}/answers_val_.pt"),
                (f"{folder}/answers_val_.pt", f"{folder}/answers_val_val.pt", f"{folder}/answers_val_.pt"),
                (f"{folder}/answers_val_.pt", f"{folder}/answers_val_only_val.pt", f"{folder}/answers_val.pt"),
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
    sweep = sweep or os.environ.get("SWEEP", "extended_tasks")
    pretrain_model = pretrain_model or os.environ.get("PRETRAIN_MODEL", "Salesforce/codegen-350m-mono")
    PORT = port or PORT

    print(f"{DATASET_KIND=} {FAKE=} {sweep=} {pretrain_model=}")
    print("GPUs available:", torch.cuda.device_count())

    if DATASET_KIND == "diamonds":
        # seeds = list(range(2))
        # seeds = list(range(4))
        seeds = list(range(8))
        generate(pretrain_model, seeds)
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
