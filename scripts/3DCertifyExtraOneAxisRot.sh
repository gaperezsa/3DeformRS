CUDA_VISIBLE_DEVICES=3 python3 Compare3DCertify.py --sigma 0.025 --certify_method rotationX --uniform --experiment_name 64pointnetRotationX0.025
CUDA_VISIBLE_DEVICES=3 python3 Compare3DCertify.py --sigma 0.075 --certify_method rotationX --uniform --experiment_name 64pointnetRotationX0.075
CUDA_VISIBLE_DEVICES=3 python3 Compare3DCertify.py --sigma 0.025 --certify_method rotationY --uniform --experiment_name 64pointnetRotationY0.025
CUDA_VISIBLE_DEVICES=3 python3 Compare3DCertify.py --sigma 0.075 --certify_method rotationY --uniform --experiment_name 64pointnetRotationY0.075
