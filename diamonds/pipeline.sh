#!/bin/bash

# Data generation

seed=${SEED:-1}
aggr_method=${AGGR_METHOD:-last}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export DATASET_GEN_MODE=with_false_negs
num_processes=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# will still generate if data not found 
DO_GENERATION=0
# DO_PRETRAIN=0
DO_TRAIN=1
DO_EVAL=1
export FORCE_EVALUATE=${DO_EVAL}

export DATASET_VERSION=${DATASET_VERSION:-3}
folder_name="v${DATASET_VERSION}.5"

tiny_model="Salesforce/codegen-350m-mono"
small_model="Salesforce/codegen-2B-mono"
base_model_folder="~/datasets/elk/diamonds/${folder_name}/models"
model_folder="~/datasets/elk/diamonds/${folder_name}/models/no_pt_${aggr_method}_s${seed}"
data_folder="~/datasets/elk/diamonds/${folder_name}/data/many_answers_s${seed}"
script_folder="./elk/func_correct/diamonds"
eval_seed=$((seed + 1000))
epochs=10

# pretrain_model="${base_model_folder}/tiny_model/end_state/save_pretrained"
pretrain_model=$tiny_model

python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_tiny_test0.pt --n 500 --tokenizer_name $tiny_model --min_tamper 5 --min_true_pos 5 --min_full_neg 5 --seed 0 --split_seed 0
python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_tiny_test1.pt --n 500 --tokenizer_name $tiny_model --min_tamper 5 --min_true_pos 5 --min_full_neg 5 --seed 0 --split_seed 0
echo "Done generating data"
exit 1

# n_ntp=100000
# n_answers=5000
# n_ntp=0
n_answers=25000

prop_clean=0.2
# n_ntp_clean=$(awk "BEGIN {print int($prop_clean * $n_ntp)}")
# n_ntp_dirty=$(awk "BEGIN {print int((1 - $prop_clean) * $n_ntp)}")
n_answers_clean=$(awk "BEGIN {print int($prop_clean * $n_answers)}")
n_answers_dirty=$(awk "BEGIN {print int((1 - $prop_clean) * $n_answers)}")

gen_args="--split_seed $seed --seed $seed --tokenizer_name $tiny_model"
val_gen_args="--split_seed $seed --seed $eval_seed --tokenizer_name $tiny_model"

clean_gen_args="$gen_args --min_prop_true_pos 0.5 --min_prop_full_neg 0.5 --difficulty easy --skip_mehs"
dirty_gen_args="$gen_args --min_prop_tamper 0.05 --min_prop_true_pos 0.4 --min_prop_full_neg 0.35 --difficulty both"
gt_gen_args="$gen_args --min_prop_tamper 0.4 --min_prop_true_pos 0.4 --min_prop_full_neg 0.1 --skip_mehs --difficulty both"

clean_val_args="$val_gen_args --min_prop_true_pos 0.8 --min_prop_full_neg 0.2 --difficulty easy --skip_mehs"
gt_val_gen_args="$val_gen_args --min_prop_tamper 0.4 --min_prop_true_pos 0.4 --min_prop_full_neg 0.1 --skip_mehs"
ntp_val_gen="$val_gen_args --pretrain --skip_mehs"

function merge_data() {
    python ${script_folder}/merge.py ${data_folder}/${1}.pt ${data_folder}/${2}.pt ${data_folder}/${3}.pt --seed $seed
}

# check if answers_val.pt exists and is not set to be regenerated
if [ ! -f "${data_folder/#\~/$HOME}/answers_val.pt" ] || [ $DO_GENERATION -eq 1 ]; then
    echo "Data not found, generating..."
    DO_GENERATION=1
fi

if [ $DO_GENERATION -eq 1 ]; then
    # python ${script_folder}/train_data_gen.py --save_path ${data_folder}/ntp_train_easy.pt --n $n_ntp_clean $clean_gen_args --pretrain
    # merge_data answers_train_easy ntp_train_easy train_easy
    # python ${script_folder}/train_data_gen.py --save_path ${data_folder}/ntp_train_both.pt --n $n_ntp_dirty $dirty_gen_args --pretrain

    python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_train_easy.pt --n $n_answers_clean $clean_gen_args
    python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_train_both.pt --n $n_answers_dirty $dirty_gen_args
    merge_data answers_train_easy answers_train_both answers_train

    python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_train_gt_both.pt --n $n_answers_dirty $gt_gen_args

    # merge_data answers_train_both ntp_train_both train_both
    # merge_data train_easy train_both train
    # merge_data ntp_train_easy ntp_train_both ntp_train

    python ${script_folder}/train_data_gen.py --save_path ${data_folder}/ntp_val.pt --n 1000 $ntp_val_gen
    python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_val_easy.pt --n 1000 $clean_val_args
    python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_val_both.pt --n 2000 $gt_val_gen_args --difficulty both
    python ${script_folder}/train_data_gen.py --save_path ${data_folder}/answers_val_val.pt --n 3000 $gt_val_gen_args --difficulty val
    merge_data answers_val_easy answers_val_both answers_val_
    merge_data answers_val_ answers_val_val answers_val
fi

always_args="--num_warmup_steps 64 --mixed_precision fp16 --dataset_kind diamonds --load_dir $data_folder --aggregation_mode $aggr_method --dont_save_state --seed 0 --checkpointing_steps 3_per_epoch --with_tracking"

tiny_args="--per_device_train_batch_size 32 --per_device_eval_batch_size 8 --num_train_epochs $epochs $always_args"
tiny_args_freeze="--learning_rate 2e-4 --freeze_layers all $tiny_args"
tiny_args_nofreeze="--learning_rate 2e-5 $tiny_args"

pretrain_args="--train_split ntp_train --token_loss_weight 1 --overall_loss_weight 0"

no_tok="--token_loss_weight 0 --train_split answers_train"
gt_weights="$no_tok --ground_truth_weight 1 --overall_loss_weight 1 --train_filter nano_tamper"
dirty_weights="$no_tok --ground_truth_weight 0"
really_dirty_weights="$no_tok --ground_truth_weight 0 --overall_loss_weight 1"
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
        DATASET_KIND=diamonds CUDA_VISIBLE_DEVICES=${device:-${CUDA_VISIBLE_DEVICES}} python elk/func_correct/eval_models.py run $model $epoch --load_dir $data_folder --split answers_val
    fi
}

train_and_evaluate() {
    local model=$1
    local output_dir=$2
    local train_args=$3

    if [ $DO_TRAIN -eq 1 ]; then
        accelerate launch --num_processes $num_processes elk/func_correct/train_fsdp.py --model_name_or_path $model --output_dir $output_dir $train_args
    fi
    # Evaluate models after each half epoch in parallel
    evaluate $output_dir epoch_0.0 0 && evaluate $output_dir epoch_0.1 0 &
    evaluate $output_dir epoch_0.2 1 && evaluate $output_dir epoch_1.0 1 &
    evaluate $output_dir epoch_1.1 2 && evaluate $output_dir epoch_1.2 2 &
    evaluate $output_dir epoch_2.0 3 && evaluate $output_dir epoch_2.1 3 &
    evaluate $output_dir epoch_2.2 4 && evaluate $output_dir epoch_3.0 4 &
    evaluate $output_dir epoch_3.1 5 && evaluate $output_dir epoch_3.2 5 &
    evaluate $output_dir epoch_4.0 6 && evaluate $output_dir epoch_4.1 6 &
    evaluate $output_dir epoch_4.2 7 && evaluate $output_dir end_state 7 &
    wait
    evaluate $output_dir epoch_5.0 0 && evaluate $output_dir epoch_5.1 0 &
    evaluate $output_dir epoch_5.2 1 && evaluate $output_dir epoch_6.0 1 &
    evaluate $output_dir epoch_6.1 2 && evaluate $output_dir epoch_6.2 2 &
    evaluate $output_dir epoch_7.0 3 && evaluate $output_dir epoch_7.1 3 &
    evaluate $output_dir epoch_7.2 4 && evaluate $output_dir epoch_8.0 4 &
    evaluate $output_dir epoch_8.1 5 && evaluate $output_dir epoch_8.2 5 &
    evaluate $output_dir epoch_9.0 6 && evaluate $output_dir epoch_9.1 6 &
    evaluate $output_dir epoch_9.2 7 &
    wait
}

# Train and evaluate models for 350M
# pretrain
# if [ $DO_PRETRAIN -eq 1 ]; then
#     export GENERATE_STDOUT=1
#     train_and_evaluate $tiny_model "${base_model_folder}/tiny_model" "$pretrain_args $tiny_args_nofreeze"
#     export GENERATE_STDOUT=0
# fi

# Train dirty model
dirty_model_path="${model_folder}/tiny_model_dirty/end_state/save_pretrained"
train_and_evaluate $pretrain_model "${model_folder}/tiny_model_dirty" "$dirty_weights $tiny_args_nofreeze"
train_and_evaluate $pretrain_model "${model_folder}/tiny_model_really_dirty" "$really_dirty_weights $tiny_args_nofreeze"
# The GT probe model
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_gt_probe" "$gt_weights $tiny_args_freeze"
# FT on clean
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_clean_probe" "$clean_weights $tiny_args_freeze"

# junction exclusion FT
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_jeft" "--excl_sensor_mask 0 1 1 1 1 $jeft_weights $tiny_args_nofreeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_jeft_dp" "--excluded_set dirty-positive --excl_sensor_mask 0 1 1 1 1 $jeft_weights $tiny_args_nofreeze"

# Some HP Search
train_and_evaluate $pretrain_model "${model_folder}/tiny_model_dirty_high_lr" "$dirty_weights --learning_rate 8e-5 $tiny_args"
train_and_evaluate $pretrain_model "${model_folder}/tiny_model_dirty_low_lr" "$dirty_weights --learning_rate 4e-6 $tiny_args"
train_and_evaluate $pretrain_model "${model_folder}/tiny_model_dirty_very_high_lr" "$dirty_weights --learning_rate 2e-4 $tiny_args"
train_and_evaluate $pretrain_model "${model_folder}/tiny_model_dirty_very_low_lr" "$dirty_weights --learning_rate 1e-6 $tiny_args"

train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_gt_probe_high_lr" "$gt_weights --learning_rate 8e-4 --freeze_layers all $tiny_args"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_gt_probe_low_lr" "$gt_weights --learning_rate 4e-5 --freeze_layers all $tiny_args"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_gt_probe_very_high_lr" "$gt_weights --learning_rate 2e-3 --freeze_layers all $tiny_args"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_gt_probe_very_low_lr" "$gt_weights --learning_rate 1e-5 --freeze_layers all $tiny_args"

# GT no freeze
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_gt" "$gt_weights $tiny_args_nofreeze"
# clean no freeze
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_clean" "$clean_weights $tiny_args_nofreeze"
train_and_evaluate $pretrain_model "${model_folder}/tiny_model_really_clean" "$clean_weights $tiny_args_nofreeze"

# exclusion fine-tuning
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_011" "--excl_sensor_mask 1 1 0 1 1 $excl_weights $tiny_args_nofreeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_101" "--excl_sensor_mask 1 1 1 0 1 $excl_weights $tiny_args_nofreeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_110" "--excl_sensor_mask 1 1 1 1 0 $excl_weights $tiny_args_nofreeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_011_dp" "--excl_sensor_mask 1 1 0 1 1 $excl_weights --excluded_set dirty-positive-0 $tiny_args_nofreeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_101_dp" "--excl_sensor_mask 1 1 1 0 1 $excl_weights --excluded_set dirty-positive-1 $tiny_args_nofreeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_110_dp" "--excl_sensor_mask 1 1 1 1 0 $excl_weights --excluded_set dirty-positive-2 $tiny_args_nofreeze"

# Amnesic
make_args_amnesic() {
    # replace --aggregation_mode last by --aggregation_mode amnesic_last (and assert that the replacement was made)
    local args=$1
    local amnesic_args=${args/--aggregation_mode last/--aggregation_mode amnesic_last}
    if [ "$args" == "$amnesic_args" ]; then
        echo "Error: aggregation_mode not found in args"
        exit 1
    fi
    echo $amnesic_args
}

amnesic_tiny_args_freeze=$(make_args_amnesic "$tiny_args_freeze")

train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_amnesic_dirty_probe" "$dirty_weights $amnesic_tiny_args_freeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_amnesic_gt_probe" "$gt_weights $amnesic_tiny_args_freeze"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_amnesic_clean_probe" "$clean_weights $amnesic_tiny_args_freeze"

# no-holds-bar
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm1_dirty_probe" "$dirty_weights $tiny_args_freeze --remove_after_layer -1"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm1_gt_probe" "$gt_weights $tiny_args_freeze --remove_after_layer -1"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm1_clean_probe" "$clean_weights $tiny_args_freeze --remove_after_layer -1"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm3_dirty_probe" "$dirty_weights $tiny_args_freeze --remove_after_layer -3"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm3_gt_probe" "$gt_weights $tiny_args_freeze --remove_after_layer -3"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm3_clean_probe" "$clean_weights $tiny_args_freeze --remove_after_layer -3"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm5_dirty_probe" "$dirty_weights $tiny_args_freeze --remove_after_layer -5"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm5_gt_probe" "$gt_weights $tiny_args_freeze --remove_after_layer -5"
train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_rm5_clean_probe" "$clean_weights $tiny_args_freeze --remove_after_layer -5"

train_and_evaluate $dirty_model_path "${model_folder}/tiny_model_dirty_probe" "$dirty_weights $tiny_args_freeze"
