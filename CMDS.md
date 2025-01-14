# install

pip install pywavelets

# train

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env demos/demo_classifier.py

python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py

python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py
screen python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py

torchrun --nproc_per_node=8 demos/demo_classifier.py

# demo
python demos/demo.py --dir_plot_save /home/manu/tmp/demo_arc_detector_save_single
