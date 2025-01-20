# install

pip install pywavelets

# train

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env demos/demo_classifier.py

python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py

python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py
screen python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py

torchrun --nproc_per_node=8 demos/demo_classifier.py
screen torchrun --nproc_per_node=8 demos/demo_classifier.py

# demo
python demos/demo.py --dir_plot_save /home/manu/tmp/demo_arc_detector_save_single --dtr_type DetectorWrapperV0 --db_key default --addr "/media/manu/ST8000DM004-2U91/afdd/data/data_v7/故障电弧测试数据-11.15/负载抑制性试验1 - labeled/吸尘器.npy"
