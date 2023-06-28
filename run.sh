python utils/dataset_preparation.py --raw_data_dir /data/xiucheng/oppo-transport/beijing --data_dir data_beijing

python train.py --data_dir data_beijing

#python utils/dataset_preparation.py --raw_data_dir /data/xiucheng/oppo-transport/harbin --data_dir data_harbin
#python train.py --data_dir data_harbin --max_epochs 5 --hidden_size 320 --batch_size 32 --n_class 5
