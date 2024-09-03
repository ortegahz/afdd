# train

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env demos/demo_classifier.py

CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py
