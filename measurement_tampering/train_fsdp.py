# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any, Optional, Union

import torch
from attrs import Factory, define
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (  # type: ignore
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from measurement_tampering.activations_utils import get_layers, set_transformer
from measurement_tampering.model_with_answer_pred import (
    GeneralLayerMdRemoverArgs,
    GeneralMdRemoverArgs,
    ModelWithAnswerPred,
)
from measurement_tampering.settings import get_setting
from measurement_tampering.train_utils import map_loss_and_preds

# from torch import _dynamo


# _dynamo.config.cache_size_limit = 512
# does nothing here?
torch.backends.cuda.matmul.allow_tf32 = True


# NOTE by Ryan: this is modified version of the accelerate FSDP example
# (originally for training BERT). I don't have any strong claims about how
# efficient etc. this is.

Num = Union[int, float]


def step_lerp(start: torch.Tensor, end: torch.Tensor, step: Num, steps_to_end: Num):
    if step > steps_to_end:
        return end
    weight = step / steps_to_end
    return torch.lerp(start, end, weight)


def step_cosine_decay(start: torch.Tensor, end: torch.Tensor, step: Num, steps_to_end: Num):
    return torch.lerp(
        start,
        end,
        (1 - torch.cos(step_lerp(torch.tensor(0.0).to(start), torch.tensor(math.pi).to(start), step, steps_to_end)))
        / 2,
    )


def get_cosine_func_with_warmup(
    lr_warmup_steps: int, total_steps: int, lr: float, warmup_up_start_lr_mul: float = 0.1, final_lr_mul: float = 0.1
):
    total_post_warmup_steps = total_steps - lr_warmup_steps
    warmup_up_start_lr = warmup_up_start_lr_mul * lr
    end_of_warmup_lr = lr
    final_lr = lr * final_lr_mul

    def run_func(step: int):
        is_warmup = step < lr_warmup_steps
        if is_warmup:
            return step_lerp(torch.tensor(warmup_up_start_lr), torch.tensor(end_of_warmup_lr), step, lr_warmup_steps)
        else:
            real_step = step - lr_warmup_steps
            return step_cosine_decay(torch.tensor(lr), torch.tensor(final_lr), real_step, total_post_warmup_steps)

    return lambda x: float(run_func(x))


def training_function(args):
    from accelerate import Accelerator, FullyShardedDataParallelPlugin  # type: ignore

    # Initialize accelerator
    # plugin = FullyShardedDataParallelPlugin()
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            log_with="wandb",
            project_dir=args.project_dir,
            # fsdp_plugin=plugin,
        )
        accelerator.init_trackers(args.project_name, config=vars(args))
    else:
        accelerator = Accelerator(
            cpu=args.cpu,
            mixed_precision=args.mixed_precision,
            # fsdp_plugin=plugin,
        )
    accelerator.print(accelerator.distributed_type)

    for k, v in get_config(args).items():
        accelerator.print(f"{k:>25}: {v}")

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = args.learning_rate
    num_epochs = int(args.num_train_epochs)
    seed = int(args.seed)
    batch_size = int(args.per_device_train_batch_size)

    model_name_or_path = os.path.expanduser(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    data_dir_here = os.path.expanduser(args.load_dir)

    setting = get_setting(args.dataset_kind)

    if args.seq_len is None:
        seq_len = min(tokenizer.model_max_length, 1024)
    else:
        seq_len = args.seq_len
        if seq_len > tokenizer.model_max_length:
            accelerator.print(
                f"The seq_len passed ({args.seq_len}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using seq_len={tokenizer.model_max_length}."
            )
            seq_len = tokenizer.model_max_length

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    get_losses = setting.create_get_losses(
        tokenizer,
        seq_len=seq_len,
        ground_truth_weight=args.ground_truth_weight,
        pad_left_with_dots=args.pad_left_with_dots,
        token_loss_weight=args.token_loss_weight,
        overall_loss_weight=args.overall_loss_weight,
        excl_sensor_mask=args.excl_sensor_mask,
        excluded_set=args.excluded_set,
        junction_target=args.junction_target,
    )

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1

    set_seed(seed)

    val_splits = args.val_splits or setting.val_splits
    val_datasets = {split: setting.load_data(data_dir_here, split) for split in val_splits}
    train_dataset = setting.load_data(data_dir_here, args.train_split)

    if args.train_filter is not None:
        amount_old_data = len(train_dataset)
        train_dataset = train_dataset.filter_cat(args.train_filter)
        accelerator.print(f"Filtered {args.train_filter} train data, {len(train_dataset)} / {amount_old_data} left.")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=default_data_collator)
    val_dataloaders = {
        k: DataLoader(
            v, batch_size=int(args.per_device_eval_batch_size), shuffle=False, collate_fn=default_data_collator
        )
        for k, v in val_datasets.items()
    }

    accelerator.print(f"Val splits: {list(val_dataloaders.keys())}")

    general_md_remover_args = GeneralMdRemoverArgs(args.remove_labels, args.lock_remover)
    if args.remove_after_layer is not None:
        general_layer_md_remover_args = GeneralLayerMdRemoverArgs(args.remove_after_layer, general_md_remover_args)
    else:
        general_layer_md_remover_args = None

    # see also load from fold
    if args.use_sensor_md_remover:
        sensor_md_remover_args = GeneralMdRemoverArgs(
            args.sensor_md_remover_remove_labels, args.sensor_md_remover_is_locked
        )
    else:
        sensor_md_remover_args = None

    if os.path.exists(f"{model_name_or_path}/get_sensors_state_dict.pt") and not args.re_init_probes:
        model = ModelWithAnswerPred.from_folder(
            model_name_or_path,
            get_embeds_mode=args.embeds_mode,
            get_sensors_mode=args.sensors_mode,
            general_layer_md_remover_args=general_layer_md_remover_args,
            sensor_remover_args=sensor_md_remover_args,
            bdas_after_layer=args.bdas_after_layer,
        )
        orig_model = model.model
    else:
        if (
            "cekal/mpt-7b-peft-compatible" == model_name_or_path
            or "mosaicml/mpt-7b" == model_name_or_path
            or args.is_mpt
        ):
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            config.attn_config["attn_impl"] = "triton"
            # config.init_device = "cuda:0"
            config.return_dict = True
            config.use_cache = False

            orig_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
        else:
            orig_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, return_dict=True, trust_remote_code=True, use_cache=False
            )

        model = ModelWithAnswerPred(
            orig_model,
            probe_dim=setting.nb_overall_sensor_values,
            get_embeds_mode=args.embeds_mode,
            get_sensors_mode=args.sensors_mode,
            general_layer_md_remover_args=general_layer_md_remover_args,
            sensor_remover_args=sensor_md_remover_args,
            bdas_after_layer_with_n_sensors=(
                (args.bdas_after_layer, setting.nb_individual_sensor_values)
                if args.bdas_after_layer is not None
                else None
            ),
            seq_len=setting.seq_len - 1,
        )
        if not args.random_probe_init:
            model.init_to_toks(*setting.initialization_tokens(tokenizer))

    # torch.autograd.detect_anomaly(check_nan=False)

    # note: requires https://github.com/huggingface/transformers/pull/21979 to
    # work properly. I think I needed nightly at the time this is written.
    if not args.no_grad_checkpointing:
        orig_model.gradient_checkpointing_enable()

    if args.freeze_layers == "all":
        model.model.requires_grad_(False)
    elif args.freeze_layers.startswith("last"):
        k = int(args.freeze_layers.removeprefix("last"))
        for layer in list(get_layers(model.model))[-k:]:
            layer.requires_grad_(False)
    elif args.freeze_layers.startswith("exceptlast"):
        k = int(args.freeze_layers.removeprefix("exceptlast"))
        for layer in list(get_layers(model.model))[:-k]:
            layer.requires_grad_(False)
    elif args.freeze_layers != "none":
        raise ValueError(f"Unknown freeze_layers: {args.freeze_layers}")

    def params_to_sync():
        # manual sync of trainable params
        # not included in fsdp because these params need to be frozen separately
        return list(model.get_embeds_for_sensors.parameters()) + list(model.get_sensors.parameters())

    if args.freeze_get_embeds:
        model.get_embeds_for_sensors.requires_grad_(False)
    if args.freeze_get_sensors:
        model.get_sensors.requires_grad_(False)

    # doesn't work for fsdp reasons
    # if args.token_loss_weight == 1:
    #     model.get_answer.requires_grad_(False)  # avoid collapsing to 0

    # New Code #
    # For FSDP feature, it is highly recommended and efficient to prepare the model before creating optimizer
    set_transformer(model.model, accelerator.prepare(model.transformer))

    # Instantiate optimizer
    # New Code #
    # For FSDP feature, at present it doesn't support multiple parameter groups,
    # so we need to create a single parameter group for the whole model
    params = [p for p in model.parameters() if p.requires_grad]

    num_params_in_model = sum(p.numel() for p in model.parameters())
    num_params_in_optimizer = sum(p.numel() for p in params)
    num_param_trained = sum(p.numel() for p in params if p.requires_grad)
    accelerator.print(
        f"Number of parameters trained: {num_param_trained} / {num_params_in_model} ({num_params_in_optimizer} in optimizer)"
    )

    optimizer = torch.optim.AdamW(params=params, lr=lr, weight_decay=args.weight_decay)

    # Instantiate scheduler
    cosine_lr_func = get_cosine_func_with_warmup(
        lr_warmup_steps=args.num_warmup_steps, total_steps=len(train_dataloader) * num_epochs, lr=1.0
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr_func)

    # New Code #
    # For FSDP feature, prepare everything except the model as we have already prepared the model
    # before creating the optimizer
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    (
        optimizer,
        train_dataloader,
        *l,
        lr_scheduler,
        # model,
    ) = accelerator.prepare(
        optimizer,
        train_dataloader,
        *val_dataloaders.values(),
        lr_scheduler,
        # model,
    )

    for k, v in zip(val_dataloaders.keys(), l):
        val_dataloaders[k] = v

    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch" or args.checkpointing_steps.startswith("every_"):
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        elif args.checkpointing_steps.endswith("_per_epoch"):
            _ = len(train_dataloader) // int(args.checkpointing_steps.removesuffix("_per_epoch"))
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps == "none":
            pass
        else:
            raise ValueError(f"Invalid argument `checkpointing_steps`: {args.checkpointing_steps}` passed.")
    else:
        checkpointing_steps = None

    def save_model(raw_output_dir: str, just_pretrained: bool = False):
        if args.output_dir is not None:
            args_output_dir = os.path.expanduser(args.output_dir)
        else:
            args_output_dir = None

        if args_output_dir is not None:
            output_dir = os.path.join(args_output_dir, raw_output_dir)
        else:
            output_dir = raw_output_dir
        if not just_pretrained and not args.dont_save_state:
            accelerator.save_state(output_dir)

        if args_output_dir is not None:
            pretrained_output_dir = os.path.join(args_output_dir, raw_output_dir, "save_pretrained")
        else:
            pretrained_output_dir = os.path.join(raw_output_dir, "save_pretrained")

        os.makedirs(output_dir, exist_ok=True)

        # save config
        json.dump(get_config(args), open(os.path.join(output_dir, "config.json"), "w"))

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        full_state_dict = accelerator.get_state_dict(model)
        pretrain_state_dict = {
            k.removeprefix("model."): t for k, t in full_state_dict.items() if k.startswith("model.")
        }
        get_embeds_dict = {
            k.removeprefix("get_embeds."): t for k, t in full_state_dict.items() if k.startswith("get_embeds.")
        }
        get_sensors_dict = {
            k.removeprefix("get_sensors."): t for k, t in full_state_dict.items() if k.startswith("get_sensors.")
        }
        md_remover_dict = {
            k.removeprefix("md_remover."): t for k, t in full_state_dict.items() if k.startswith("md_remover.")
        }
        sensor_md_remover_dict = {
            k.removeprefix("sensor_md_remover."): t
            for k, t in full_state_dict.items()
            if k.startswith("sensor_md_remover.")
        }
        bdas_dict = {k.removeprefix("bdas."): t for k, t in full_state_dict.items() if k.startswith("bdas.")}

        # New Code #
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.model.save_pretrained(
            pretrained_output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=pretrain_state_dict,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(pretrained_output_dir)
            torch.save(get_embeds_dict, f"{pretrained_output_dir}/get_embeds_state_dict.pt")
            torch.save(get_sensors_dict, f"{pretrained_output_dir}/get_sensors_state_dict.pt")
            if sensor_md_remover_dict:
                torch.save(sensor_md_remover_dict, f"{pretrained_output_dir}/sensor_md_remover_state_dict.pt")

            if args.remove_after_layer is not None:
                assert md_remover_dict
                torch.save(md_remover_dict, f"{pretrained_output_dir}/md_remover_state_dict.pt")

            if args.bdas_after_layer is not None:
                assert bdas_dict
                torch.save(bdas_dict, f"{pretrained_output_dir}/bdas_state_dict.pt")

    def evaluate(epoch_str: str, train_total_metrics: Optional[dict] = None):
        model.eval()

        @torch.no_grad()
        def compute_metrics(data_loader, split_name):
            total_metrics = defaultdict(int)

            def it():
                for batch in tqdm(data_loader, desc=f"evaluating {split_name}"):
                    loss_and_preds = get_losses(model, batch)
                    loss_and_preds = map_loss_and_preds(
                        lambda t: accelerator.gather_for_metrics(t.detach().contiguous()).cpu(), loss_and_preds
                    )
                    for k, v in loss_and_preds.metrics.items():
                        total_metrics[k] += float(v.mean())
                    yield {
                        k: accelerator.gather_for_metrics(t.detach().contiguous()).cpu() for k, t in batch.items()
                    }, loss_and_preds

            r = concat_and_save(it(), tokenizer, args.load_dir, args.output_dir, epoch_str, args.tiny_run, split_name)

            dirty_positive = r["all_passes"] * (~r["is_clean"])
            is_correct_dp = r["is_correct"][dirty_positive]
            all_passes_scores_dp = r["answer_all_passes"][dirty_positive]
            answer_scores_dp = r["answer_correct"][dirty_positive]

            try:
                auroc_tns = {
                    "all": roc_auc_score(is_correct_dp, all_passes_scores_dp),
                    "correct": roc_auc_score(is_correct_dp, answer_scores_dp),
                }
            except:
                auroc_tns = None  # can fail when not enough datapoints

            return total_metrics, auroc_tns

        def print_and_log_str(name, dataloader, total_metrics, auroc_tns=None):
            avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}

            metric_strs = " ".join(f"{k}={avg:.5f}" for k, avg in avg_metrics.items())

            auroc_tn_suffix = (
                " ".join(f"auroc_tn_{k}={auroc}" for k, auroc in auroc_tns.items()) if auroc_tns is not None else ""
            )

            to_log = {f"{name}/avg_{k}": v for k, v in avg_metrics.items()}
            if auroc_tns is not None:
                to_log |= {f"{name}/auroc_tn_{k}": v for k, v in auroc_tns.items()}
            if args.with_tracking:
                accelerator.log(to_log)

            return f"{name} [{metric_strs} {auroc_tn_suffix}]"

        to_print = (
            [] if train_total_metrics is None else [print_and_log_str("train", train_dataloader, train_total_metrics)]
        ) + [print_and_log_str(k, v, *compute_metrics(v, k)) for k, v in val_dataloaders.items()]
        accelerator.print(*to_print)

    if args.eval_and_exit:
        evaluate("end_state")
        save_model("end_state")
        return

    if args.save_and_exit:
        save_model("save_and_exit", just_pretrained=True)
        return

    overall_step = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(len(train_dataloader) * num_epochs),
        disable=not accelerator.is_local_main_process,
    )

    @define
    class ValueTracker:
        exp_alpha: float = 0.999
        items: list[float] = Factory(list)
        exp_moving_average_raw: float = 0.0

        def append(self, item: float):
            self.items.append(item)
            self.exp_moving_average_raw = self.exp_alpha * self.exp_moving_average_raw + (1 - self.exp_alpha) * item

        @property
        def exp_moving_average(self):
            return self.exp_moving_average_raw / (1 - self.exp_alpha ** len(self.items))  # debias our estimate

    metric_tracks = defaultdict(lambda: ValueTracker(0.9))

    # Train
    for epoch in range(num_epochs):
        model.train()
        total_metrics = defaultdict(int)
        for step, batch in enumerate(train_dataloader):
            if args.tiny_run and step > 2:
                break

            total_loss, metrics, *_ = get_losses(model, batch)
            total_loss = total_loss / gradient_accumulation_steps
            # We keep track of the loss at each epoch
            accelerator.backward(total_loss)

            # manually sync
            for p in params_to_sync():
                assert p.grad is not None
                p.grad[:] = accelerator.gather(p.grad[None]).mean(dim=0)

            for k, v in metrics.items():
                total_metrics[k] += float(accelerator.gather_for_metrics(v.detach()).mean())

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # manually sync
            for p in params_to_sync():
                p.data[:] = accelerator.gather(p.data[None]).mean(dim=0)

            for k, v in metrics.items():
                metric_tracks[k].append(v.item())

            if args.with_tracking:
                accelerator.log({k: v.item() for k, v in metrics.items()})

            if step % gradient_accumulation_steps == 0:
                progress_bar.set_postfix(
                    {k: f"{v.exp_moving_average:.5f}" for k, v in metric_tracks.items()}
                    | {"lr": lr_scheduler.get_last_lr()}
                )

            if model.bdas is not None:
                model.bdas.update_temp(overall_step / (len(train_dataloader) * num_epochs))

            overall_step += 1

            if isinstance(checkpointing_steps, int):
                if overall_step % checkpointing_steps == 0:
                    save_model(f"step_{overall_step}")
            elif checkpointing_steps.endswith("_per_epoch"):
                n_checkpoints_per_epoch = int(checkpointing_steps.removesuffix("_per_epoch"))
                checkpointing_steps_ = len(train_dataloader) // n_checkpoints_per_epoch
                q, r = divmod(step, checkpointing_steps_)
                if r == 0 and q < n_checkpoints_per_epoch:
                    save_model(f"epoch_{epoch}.{q}")

        epoch_str = f"epoch_{epoch}" if epoch < num_epochs - 1 else "end_state"
        evaluate(epoch_str, total_metrics)

        if checkpointing_steps == "epoch":
            save_model(f"epoch_{epoch}")
        if checkpointing_steps.startswith("every_"):
            every_n_epoch = int(checkpointing_steps.split("_")[1])
            if (epoch + 1) % every_n_epoch == 0 and epoch < num_epochs - 1:
                save_model(f"epoch_{epoch}")

    save_model(f"end_state")

    if args.with_tracking:
        accelerator.end_training()


def get_parser():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help=(
            "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch,"
            " or '{n}_per_epoch' for n times per epoch, or 'every_{n}_epoch for every n epochs, or 'none'"
        ),
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Optional save directory where all checkpoint folders will be stored. Default is the current working directory.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs`",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens). func_correct only"
        ),
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=64,
    )
    parser.add_argument("--weight_decay", type=float, default=0.02, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--save_and_exit", action="store_true")
    parser.add_argument("--eval_and_exit", action="store_true")
    parser.add_argument(
        "--token_loss_weight", type=float, default=0.15, help="fraction of loss which is used for next token prediction"
    )
    parser.add_argument(
        "--overall_loss_weight",
        type=float,
        default=0.2,
        help="fraction of loss which is used for all passes and is correct",
    )
    parser.add_argument(
        "--ground_truth_weight",
        type=float,
        default=0.5,
        help="weight of ground truth in overall loss, 0.0 means no ground truth",
    )
    parser.add_argument(
        "--excl_sensor_mask",
        type=float,
        nargs="*",
        default=None,
        help="mask sensors in sensor loss for exclusion FT, only used on dirty & half-neg. diamonds only",
    )
    parser.add_argument(
        "--train_filter",
        type=str,
        default=None,
        help="filter to apply to train data, as arg to a dataset filter_cat",
    )
    parser.add_argument(
        "--pad_left_with_dots",
        action="store_true",
        help="Pad left with dots. Align answers to the right. Required for coco. func_correct only.",
    )
    parser.add_argument(
        "--tiny_run",
        action="store_true",
        help="Only train for a few steps (for debugging)",
    )
    parser.add_argument(
        "--dont_save_state",
        action="store_true",
        help="Disable saving full state (for recovering training, but takes space on disk)",
    )
    parser.add_argument(
        "--no_grad_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default="~/code_elk_setting/correctness_data/full_p_extended",
        help="Directory from which the files will be loaded. Don't include the final /",
    )
    parser.add_argument(
        "--dataset_kind",
        type=str,
        default="func_correct",
        help="Which kind of dataset & sensors to use.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="finetune",
        help="Which kind of dataset & sensors to use.",
    )
    parser.add_argument(
        "--freeze_layers",
        type=str,
        default="none",
        help="Which layers to freeze. Options: none, all, last{k}, exceptlast{k}",
    )
    parser.add_argument(
        "--freeze_get_embeds",
        action="store_true",
        help="Whether to freeze the the approach for getting embeds.",
    )
    parser.add_argument(
        "--freeze_get_sensors",
        action="store_true",
        help="Whether to freeze the the approach for converting embeds to sensor logits.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Which split to use for training.",
    )
    parser.add_argument(
        "--val_splits",
        default=None,
        type=str,
        nargs="*",
        help="Which split to use for validation.",
    )
    parser.add_argument(
        "--embeds_mode",
        type=str,
        default="identity",
        help="how to gather embeds",
    )
    parser.add_argument(
        "--sensors_mode",
        type=str,
        default="linear",
        help="how to convert embeds into logits",
    )
    parser.add_argument(
        "--excluded_set",
        type=str,
        default="dirty",
        help="What should be excluded in exclusion fine-tuning. Either dirty or dirty-positive",
    )
    parser.add_argument(
        "--remove_after_layer",
        type=int,
        default=None,
        help="After which layer to remove the information. If None, don't remove.",
    )
    parser.add_argument(
        "--remove_labels",
        type=str,
        default="is_correct",
        help="Which information to remove.",
    )
    parser.add_argument(
        "--lock_remover",
        action="store_true",
        help="Whether to keep updating the running mean of the remover.",
    )
    parser.add_argument(
        "--use_sensor_md_remover",
        action="store_true",
        help="Whether to use the mean diff remover for sensor extraction",
    )
    parser.add_argument(
        "--sensor_md_remover_remove_labels",
        type=str,
        default="all_passes",
        help="Which information to remove.",
    )
    parser.add_argument(
        "--sensor_md_remover_is_locked",
        action="store_true",
        help="Whether to keep updating the running mean of the remover.",
    )
    parser.add_argument(
        "--bdas_after_layer",
        type=int,
        default=None,
        help="After which layer to inject the bdas layer.",
    )
    parser.add_argument(
        "--is_mpt",
        action="store_true",
        help="is mpt style model",
    )
    parser.add_argument(
        "--re_init_probes",
        action="store_true",
        help="wipe probes back (as well as anything which isn't the base model, including md_remover, bdas ...)",
    )
    parser.add_argument(
        "--random_probe_init",
        action="store_true",
        help="Initialize the probes randomly instead of using the unembedded tokens.",
    )
    parser.add_argument(
        "--junction_target",
        type=str,
        default="all_passes",
        help="Which key of the batch data to use as the target for the junction loss. tamper_detection creates the field on the fly.",
    )
    return parser


def get_default_config():
    arg_defaults = {}

    # Iterate through the actions in the ArgumentParser
    for action in get_parser()._actions:
        # If the action has a default value, store it in the dictionary
        if action.default is not None:
            arg_defaults[action.dest] = action.default
        # If there is no default value, store None in the dictionary
        else:
            arg_defaults[action.dest] = None

    return arg_defaults


def dict_to_args(d: dict[str, Any]):
    class Empty:
        pass

    args = Empty()
    for k, v in d.items():
        setattr(args, k, v)
    return args


def get_config(args) -> dict[str, Any]:
    config = get_default_config()
    config.update(vars(args))
    return config


def main():
    parser = get_parser()
    args = parser.parse_args()
    training_function(args)


SPLIT_EXTENSIONS = {
    "train": ".train",
    "non_overlapping_val": "",
    "answers_val": "",
    "answers_train": ".answers_train",
    "answers_val_train": ".train_val",
    "val": "",
    "overlapping_val": ".overlapping_val",
    "ntp_val": ".ntp",
}

for k in list(SPLIT_EXTENSIONS.keys()):
    for i in range(1, 8):
        x = 2**i
        SPLIT_EXTENSIONS[f"{k}_o{x}"] = f".{k}_o{x}"


def get_path(
    load_dir: str,
    model_folder: str,
    epoch: str = "end_state",
    tmp: bool = False,
    split: str = "non_overlapping",
    max_points: Optional[int] = None,
):
    clean_load_dir = "-".join(load_dir.split("/")[-2:])

    max_points_suffix = f"_{max_points}" if max_points is not None else ""

    return (
        f"{model_folder}/{epoch}/scores_{clean_load_dir}{max_points_suffix}.pt"
        + (".tmp" if tmp else "")
        + SPLIT_EXTENSIONS[split]
    )


def concat_and_save(batch_and_pred_it, tokenizer, load_dir, model_folder, epoch, tmp, split, max_points=None):
    to_save = defaultdict(list)
    for i, (item, loss_and_preds) in enumerate(batch_and_pred_it):
        to_save["total_loss"].append(loss_and_preds.loss)
        for name, value in loss_and_preds.metrics.items():
            to_save[name].append(value)
        to_save["sensor_logits"].append(loss_and_preds.sensor_values.cpu())
        to_save["answer_all_passes"].append(loss_and_preds.all_passes_value.cpu())
        to_save["answer_correct"].append(loss_and_preds.is_correct_value.cpu())
        for k, v in item.items():
            to_save[k].append(v)
        texts = [tokenizer.decode(ids) for ids in item["input_ids"]]
        to_save["text"].append(texts)

        if tmp and i > 5:
            break

    def cat(x):
        if isinstance(x[0], torch.Tensor) and x[0].dim() == 0:
            return torch.stack(x)
        elif isinstance(x[0], torch.Tensor):
            return torch.cat(x)
        elif isinstance(x[0], list):
            return sum(x, [])
        else:
            return x

    concatenated = {k: cat(v) for k, v in to_save.items()}

    path = get_path(load_dir, model_folder, epoch, tmp, split, max_points)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(concatenated, path)

    return concatenated


if __name__ == "__main__":
    main()

# %%
