## Dataset description
The raw data is saved in `raw_data_dir` as a collection of csv files, e.g., 
```
beijing/
├── beijing-airplane-a.csv
├── beijing-airplane-b.csv
├── beijing-bus.csv
├── beijing-car-ele.csv
├── beijing-car-mix.csv
├── beijing-car-oil.csv
├── beijing-subway.csv
├── beijing-train-a.csv
└── beijing-train-b.csv
```
and `utils/dataset_preparation.py` will transform the raw csv files into the `data_dir` (e.g., `data_beijing/` of the current directory) as `trn.json`, `val.json`, `tst.json`. The dataset is split into disjoint sequences by using the `seq_id` provided in the raw csv files, and thus the `seq_id` column should be unique across all csv files (be able to identify a sequence across csv files).

## Code running

Run `python utils/dataset_preparation.py --raw_data_dir /data/xiucheng/oppo-transport/beijing --data_dir data_beijing` to generate `data_beijing/*.json`.

Run `python train.py` to train and do validation,  as well as test with the best model.