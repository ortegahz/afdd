# missing case

# install
pip install pywavelets

# data
python utils/svm2hdf5.py

# train

export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env demos/demo_classifier.py

python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py

python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py
screen python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py
python -m torch.distributed.launch --nproc_per_node=1 --use_env demos/demo_classifier.py --save_dir /home/manu/tmp/afdd_models

torchrun --nproc_per_node=8 demos/demo_classifier.py
screen torchrun --nproc_per_node=8 demos/demo_classifier.py
torchrun --nproc_per_node=8 --master_addr=172.20.254.132 --master_port=29501 demos/demo_classifier.py

# demo
python demos/demo.py --dir_plot_save /home/manu/tmp/demo_arc_detector_save_single --dtr_type DetectorWrapperV0 --db_key default --addr "/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v14/data_pick/neg/多负载运行（电机+日光灯）2.npy"

python demos/demo.py --dir_plot_save /home/manu/tmp/demo_arc_detector_save_debug --dtr_type DetectorWrapperV3NPY --db_key default --addr "/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v14/tmp/"