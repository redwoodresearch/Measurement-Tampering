tiny_args_freeze="azesqd --aggregatio_mode last --freeze_layers all --learning_rate 2e-4 --max_epochs 1 --max_steps 100 --model_name tiny --model_type tiny --num_workers 0 --optimizer adam --pretrain --pretrain_model_name tiny --pretrain_model_type tiny --pretrain_num_workers 0 --pretrain_optimizer adam --pretrain_scheduler linear --pretrain_train_batch_size 1 --pretrain_val_batch_size 1 --scheduler linear --train_batch_size 1 --val_batch_size 1 --warmup_steps 0 --weight_decay 0.0 --window_size 1"

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

amnesic_tiny_args_freeze=$(make_args_amnesic "${tiny_args_freeze}")
echo $amnesic_tiny_args_freeze
