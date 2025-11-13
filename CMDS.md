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

python demos/demo_classifier.py
torchrun --nproc_per_node=8 demos/demo_classifier.py
screen torchrun --nproc_per_node=8 demos/demo_classifier.py
torchrun --nproc_per_node=8 --master_addr=172.20.254.132 --master_port=29501 demos/demo_classifier.py
torchrun --nproc_per_node=8 --master_addr=172.20.254.199 --master_port=29501 demos/demo_classifier.py

screen torchrun --nproc_per_node=8 --master_addr=172.20.254.199 --master_port=29501 demos/demo_classifier.py

torchrun --nproc_per_node=8 --master_addr=172.20.254.199 --master_port=29501 demos/demo_classifier.py --load_dir /home/Huangzhe/test/afd_pm_hdf5  --save_dir /home/Huangzhe/test/afdd_models_mp --path_ckpt /home/Huangzhe/test/afdd_models_mp_v5/ae_best_e680_acc0.9818.pt

# demo
python demos/demo.py --dir_plot_save /home/manu/tmp/demo_arc_detector_save_single --dtr_type DetectorWrapperV0 --db_key default --addr "/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v21/data_sorted/自主正例/电钻最高速运行_0.npy"

python demos/demo.py --dir_plot_save /home/manu/tmp/demo_arc_detector_save_debug --dtr_type DetectorWrapperV3NPY --db_key default --addr "/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v16/data_pick/pos/"

python demos/demo.py --dir_plot_save /home/manu/tmp/demo_arc_detector_save_debug --dtr_type DetectorWrapperV0 --db_key default --dbo_type DataV6 --addr "/media/manu/ST8000DM004-2U91/afdd/data/data_v26/data_sorted - o/负载抑制 - o/电容滤波器抑制试验+1000W阻性负载_20k.bin"

# others
dd if=/dev/ttyACM1 of=/home/manu/tmp/raw.bin bs=128K status=progress
sudo stty -F /dev/ttyACM1 raw -echo -ixon -ixoff -crtscts
dd if=/dev/ttyACM1 of=/dev/shm/raw.bin bs=1K count=1K iflag=fullblock status=progress
INFO:root:best val_accuracy -> 0.9732620320855615
INFO:root:best val_accuracy -> 0.9732620320855615

# onnx
python -m onnxruntime.quantization.preprocess \
       --input  /home/manu/tmp/afdd_e447.onnx \
       --output /home/manu/tmp/afdd_e447_pre.onnx

# statics
INFO:root:Namespace(path_label_train='/home/Huangzhe/test/afd_pm_train', save_dir='/dev/shm/afd_pm_hdf5')
INFO:root:ros -> Counter({0: 6229475, 1: 6229475})
INFO:root:Counter(y_train_aug) -> Counter({1: 4360633, 0: 4360632})
INFO:root:Counter(y_test_aug) -> Counter({0: 1868843, 1: 1868842})
INFO:root:Counter(y_train) -> Counter({0: 4360632, 1: 97882})
INFO:root:Counter(y_test) -> Counter({0: 1868843, 1: 41949})
