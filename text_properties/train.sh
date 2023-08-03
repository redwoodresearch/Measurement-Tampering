export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --main_process_port 29516 --num_processes 4 --config_file ~/pythia_config.yaml elk/func_correct/train_fsdp.py  --model_name_or_path "EleutherAI/pythia-1.4b-deduped" --seq_len 1536 --output_dir "./pythia_1_b_far_more_data" --learning_rate 6e-6 --num_warmup_steps 32 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --num_train_epochs 4 --mixed_precision fp16 --weight_decay 0.02 --token_loss_weight 1.0 --overall_loss_weight 0.0 --dataset_kind text_properties --load_dir ~/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_mask_prefix_weighted_new_big_with_omit/ --dont_save_state
accelerate launch --main_process_port 29516 --num_processes 4 --config_file ~/pythia_config.yaml elk/func_correct/train_fsdp.py  --model_name_or_path "EleutherAI/pythia-1.4b-deduped" --seq_len 1536 --output_dir "./pythia_1_b_far_more_data_mod_explict_change_weight" --learning_rate 6e-6 --num_warmup_steps 32 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --num_train_epochs 4 --mixed_precision fp16 --weight_decay 0.02 --token_loss_weight 1.0 --overall_loss_weight 0.0 --dataset_kind text_properties --load_dir ~/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_mask_prefix_change_weighted_new_big_mod_explicit_with_omit/ --dont_save_state



export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch --main_process_port 29518 --num_processes 4 --config_file ~/pythia_config.yaml elk/func_correct/train_fsdp.py  --model_name_or_path "EleutherAI/pythia-1.4b-deduped" --seq_len 1536 --output_dir "./pythia_1_b_far_more_data_mod_sensor_pred" --learning_rate 6e-6 --num_warmup_steps 32 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --num_train_epochs 4 --mixed_precision fp16 --weight_decay 0.02 --token_loss_weight 1.0 --overall_loss_weight 0.0 --dataset_kind text_properties --load_dir ~/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_upd_with_omit_sensor_pred/ --dont_save_state

export CUDA_VISIBLE_DEVICES=4,5

accelerate launch --main_process_port 29504 --num_processes 2 --config_file ~/pythia_config.yaml elk/func_correct/train_fsdp.py  --model_name_or_path "EleutherAI/pythia-410m-deduped" --seq_len 1536 --output_dir "./pythia_400_m_far_more_data" --learning_rate 6e-6 --num_warmup_steps 8 --per_device_train_batch_size 32 --per_device_eval_batch_size 8 --num_train_epochs 4 --mixed_precision fp16 --weight_decay 0.02 --token_loss_weight 1.0 --overall_loss_weight 0.0 --dataset_kind text_properties --load_dir ~/text_properties/simplified_setting_v2/data_gpt_neo_x_tokenizer_mask_prefix_weighted_new_big_with_omit --dont_save_state

# LOWER:  636/636 [52:03<00:00,  4.83s/it, loss=0.65287, token_l=0.65287, lr=[6.000000089406967e-07]] train [loss=0.65113 token_l=0.65113 ] val [loss=0.72828 token_l=0.72828 ]

#
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

accelerate launch --main_process_port 29512 --num_processes 7 --config_file ~/mpt_config.yaml elk/func_correct/train_fsdp.py  --model_name_or_path "cekal/mpt-7b-peft-compatible" --seq_len 1536 --output_dir "$HOME/text_properties/simplified_setting_v3/upd_with_omit_sensor_pred/models/base_model" --learning_rate 2e-6 --num_warmup_steps 16 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --num_train_epochs 4 --mixed_precision fp16 --weight_decay 0.02 --token_loss_weight 1.0 --overall_loss_weight 0.0 --dataset_kind text_properties --load_dir ~/text_properties/simplified_setting_v3/upd_with_omit_sensor_pred/data_gpt_neo_x_tokenizer --dont_save_state

 # | 1278/2556 [2:30:10<2:27:32,  6.93s/it, loss=0.49664, token_l=0.49664, lr=[1.6512465476989747e-06]] train [loss=0.50145 token_l=0.50145 ] val [loss=0.52640 token_l=0.52640 ]
 #
 #
