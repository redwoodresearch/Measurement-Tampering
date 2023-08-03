
model_folder="~/datasets/elk/diamonds/models"
data_folder="~/datasets/elk/diamonds"

coco_args="diamonds "${model_folder}/tiny_model_dirty""
# CUDA_VISIBLE_DEVICES=7 python elk/func_correct/coco_global.py $coco_args --initial_lr 3e-5 & 
# CUDA_VISIBLE_DEVICES=1 python elk/func_correct/coco_global.py $coco_args --initial_lr 1e-4 &> /dev/null &
# CUDA_VISIBLE_DEVICES=2 python elk/func_correct/coco_global.py $coco_args --initial_lr 1e-4 --l1_strength 10 &> /dev/null &
# CUDA_VISIBLE_DEVICES=3 python elk/func_correct/coco_global.py $coco_args --initial_lr 1e-4 --l1_strength 0.1 &> /dev/null &
# CUDA_VISIBLE_DEVICES=4 python elk/func_correct/coco_global.py $coco_args --initial_lr 1e-4 --l1_strength 1 --adv_junction_method "none" &> /dev/null &
# CUDA_VISIBLE_DEVICES=5 python elk/func_correct/coco_global.py $coco_args --initial_lr 3e-4 --l1_strength 1 &> /dev/null &
CUDA_VISIBLE_DEVICES=7 python elk/func_correct/coco_global.py $coco_args --layers_config some --initial_lr 1e-5 & 
CUDA_VISIBLE_DEVICES=1 python elk/func_correct/coco_global.py $coco_args --layers_config some --initial_lr 3e-6 &> /dev/null &
CUDA_VISIBLE_DEVICES=2 python elk/func_correct/coco_global.py $coco_args --layers_config all_mods --initial_lr 1e-5 &> /dev/null &
CUDA_VISIBLE_DEVICES=3 python elk/func_correct/coco_global.py $coco_args --layers_config all_mods --initial_lr 3e-5 &> /dev/null &
CUDA_VISIBLE_DEVICES=4 python elk/func_correct/coco_global.py $coco_args --layers_config all_mods --initial_lr 1e-4 &> /dev/null &
CUDA_VISIBLE_DEVICES=5 python elk/func_correct/coco_global.py $coco_args --layers_config all_mods --initial_lr 3e-5 --l1_strength 10 &> /dev/null &
wait