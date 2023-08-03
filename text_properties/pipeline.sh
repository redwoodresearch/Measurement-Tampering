#!/bin/bash

# Data generation

# seed=${SEED:-1}
seed=283
embeds_mode=${EMBEDS_MODE:-identity}
# embeds_mode=${EMBEDS_MODE:-small_attn}

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1 #,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=2,3 #,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5 #,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=6,7 #,2,3,4,5,6,7
num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Save the original IFS
oldIFS=$IFS

# Set the IFS to comma
IFS=','

# Read the string into an array using the IFS
read -ra devices_arr <<< "$CUDA_VISIBLE_DEVICES"

# Restore the original IFS
IFS=$oldIFS

# will still generate if data not found 
DO_TRAIN=0
DO_EVAL=1
export FORCE_EVALUATE=${DO_EVAL}


small_model="./pythia_1_b_far_more_data_mod_sensor_pred/epoch_2/save_pretrained/"
medium_model="./mpt_run_lower_lr_far_sensor_pred/epoch_2/save_pretrained/"
model_folder="$HOME/text_properties/simplified_setting_v2/models/"
# data_folder="$HOME/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_post_pretrain/"
seq_len=1024
# data_folder="$HOME/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_post_pretrain_keep_pred/"
# data_folder="$HOME/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_post_pretrain_long/"
data_folder="$HOME/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_post_pretrain_with_extra_tamp_sensor_pred/"
# data_folder="$HOME/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_post_pretrain_keep_pred_from_model/"
# data_folder="$HOME/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_post_pretrain_keep_pred_from_model_low_temp/"
# seq_len=1536
eval_seed=$((seed + 1000))
epochs=6

mkdir -p $model_folder

# pretrain_model="${base_model_folder}/tiny_model/end_state/save_pretrained"
model_size="small"
extra_prefix="_sensor_pred"
pretrain_model=$small_model

# python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_tiny_test.pt --n 600 --tokenizer_name $tiny_model --min_tamper 100 --min_true_pos 100 --min_full_neg 100 --seed 1
# echo "Done generating data"
# exit 1
extra_args="--re_init_probes"  # fix reinit as needed
# extra_args=""

always_args="--num_warmup_steps 1 --mixed_precision fp16 --dataset_kind text_properties --load_dir $data_folder --embeds_mode $embeds_mode --sensors_mode linear --dont_save_state --seed $seed --with_tracking --seq_len $seq_len $extra_args"

accelerate_args="--config_file $HOME/pythia_config.yaml"

# standard_args="--per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs $epochs $always_args"
standard_args="--per_device_train_batch_size 32 --per_device_eval_batch_size 8 --num_train_epochs $epochs $always_args"
args_freeze="--learning_rate 5e-5 --freeze_layers all $standard_args"
# args_freeze="--learning_rate 1e-5 --freeze_layers all $standard_args"
# args_freeze="--learning_rate 5e-4 --freeze_layers all $standard_args"
# args_freeze="--learning_rate 5e-6 --freeze_layers all $standard_args"
# args_nofreeze="--learning_rate 2e-5 $standard_args"
args_nofreeze="--learning_rate 3e-6 $standard_args"
# args_nofreeze="--learning_rate 1e-6 $standard_args"
# args_nofreeze="--learning_rate 6e-5 $standard_args"
# args_nofreeze="--learning_rate 8e-6 $standard_args"
# args_nofreeze="--learning_rate 1e-6 $standard_args"

# standard_args="--per_device_train_batch_size 16 --per_device_eval_batch_size 8 --num_train_epochs $epochs $always_args"
# args_freeze="--learning_rate 1e-4 --freeze_layers all $standard_args"
# args_nofreeze="--learning_rate 1e-5 $standard_args"

no_tok="--token_loss_weight 0 --train_split train"
gt_weights="$no_tok --ground_truth_weight 1 --overall_loss_weight 1 --train_filter nano_tamper"
gt_more_data_weights="$no_tok --ground_truth_weight 1 --overall_loss_weight 1 --train_filter no_false_negatives"
dirty_weights="$no_tok --ground_truth_weight 0"
really_dirty_weights="$no_tok --ground_truth_weight 0 --overall_loss_weight 1"
somewhat_clean_weights="$no_tok --ground_truth_weight 0 --train_filter clean"
clean_weights="$no_tok --ground_truth_weight 0 --overall_loss_weight 1 --train_filter abs_clean"
excl_weights="$no_tok --overall_loss_weight 0"
tampd_weights="$no_tok --overall_loss_weight 1 --ground_truth_weight 0 --detect_half_tamper"
jeft_weights="$no_tok --overall_loss_weight 0.3 --ground_truth_weight 0"

# Define function to train and evaluate models
evaluate() {
    local model=$1
    local epoch=$2
    local device=$3

    if [ $DO_EVAL -eq 1 ]; then
        DATASET_KIND=text_properties CUDA_VISIBLE_DEVICES=${devices_arr[$device]} python elk/func_correct/eval_models.py run $model $epoch --load_dir $data_folder --split val
    fi
}

train_and_evaluate() {
    local model=$1
    local output_dir=$2
    local train_args=$3

    if [ $DO_TRAIN -eq 1 ]; then
        echo accelerate launch --num_processes $num_processes $accelerate_args elk/func_correct/train_fsdp.py --model_name_or_path $model --output_dir $output_dir $train_args
        accelerate launch --main_process_port 29591 --num_processes $num_processes $accelerate_args elk/func_correct/train_fsdp.py --model_name_or_path $model --output_dir $output_dir $train_args || exit 1
    fi
    # evaluate $output_dir epoch_1 0 &
    # evaluate $output_dir epoch_2 1 &
    # evaluate $output_dir epoch_3 2 &
    # evaluate $output_dir epoch_4 3 &
    # wait
    # evaluate $output_dir epoch_5 0 &
    # evaluate $output_dir end_state 1 &
    # wait

    evaluate $output_dir epoch_1.0 1 &
    evaluate $output_dir epoch_2.0 0 &
    wait
    evaluate $output_dir epoch_3.0 0 &
    evaluate $output_dir epoch_4.0 1 &
    wait
    evaluate $output_dir epoch_5.0 0 &
    evaluate $output_dir end_state 1 &
    wait
}

# Train dirty model
dirty_start_model_path="$pretrain_model"
# dirty_start_model_path="/home/ubuntu/text_properties/simplified_setting_v2/models/small_model_dirty_attn/epoch_4.0/save_pretrained/"

# train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_model" "$dirty_weights $args_nofreeze"
# train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_model_really_dirty" "$really_dirty_weights $args_nofreeze"

# dirty probe only
# train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_dirty_probe_only_model" "$dirty_weights $args_freeze"

# # The GT probe model
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt_probe" "$gt_weights $args_freeze"
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt_more_data_probe" "$gt_more_data_weights $args_freeze"
# # FT on clean
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_clean_probe" "$clean_weights $args_freeze"
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_somewhat_clean_probe" "$somewhat_clean_weights $args_freeze"

# junction exclusion FT
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_jeft_attn" "--excl_sensor_mask 0 1 1 1 1 $jeft_weights $args_nofreeze"
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_jeft_dp_from_dirty" "--excluded_set dirty-positive --excl_sensor_mask 0 1 1 1 1 $jeft_weights $args_nofreeze"

# # # Some HP Search
# # train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_model_dirty_high_lr" "$dirty_weights --learning_rate 8e-5 $standard_args"
# # train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_model_dirty_low_lr" "$dirty_weights --learning_rate 4e-6 $standard_args"
# # train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_model_dirty_very_high_lr" "$dirty_weights --learning_rate 2e-4 $standard_args"
# # train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_model_dirty_very_low_lr" "$dirty_weights --learning_rate 1e-6 $standard_args"

# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt_probe_high_lr" "$gt_weights --learning_rate 8e-4 --freeze_layers all $standard_args"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt_probe_low_lr" "$gt_weights --learning_rate 4e-5 --freeze_layers all $standard_args"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt_probe_very_high_lr" "$gt_weights --learning_rate 2e-3 --freeze_layers all $standard_args"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt_probe_very_low_lr" "$gt_weights --learning_rate 1e-5 --freeze_layers all $standard_args"

# # GT no freeze
train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt" "$gt_weights $args_nofreeze"
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_gt_more_data" "$gt_more_data_weights $args_nofreeze"
# # clean no freeze
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_clean" "$clean_weights $args_nofreeze"
# train_and_evaluate $pretrain_model "${model_folder}/${model_size}${extra_prefix}_model_somewhat_clean" "$somewhat_clean_weights $args_nofreeze"

# # # exclusion fine-tuning
# train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_011" "--excl_sensor_mask 1 1 0 1 1 $excl_weights $args_nofreeze"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_101" "--excl_sensor_mask 1 1 1 0 1 $excl_weights $args_nofreeze"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_110" "--excl_sensor_mask 1 1 1 1 0 $excl_weights $args_nofreeze"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_011_dp" "--excl_sensor_mask 1 1 0 1 1 $excl_weights --excluded_set dirty-positive-0 $args_nofreeze"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_101_dp" "--excl_sensor_mask 1 1 1 0 1 $excl_weights --excluded_set dirty-positive-1 $args_nofreeze"
# # train_and_evaluate $dirty_start_model_path "${model_folder}/${model_size}${extra_prefix}_model_dirty_110_dp" "--excl_sensor_mask 1 1 1 1 0 $excl_weights --excluded_set dirty-positive-2 $args_nofreeze"
