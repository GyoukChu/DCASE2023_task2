# TODO

0. Set the environment

- First, create conda environment with python==3.10.12
```
pip install -r requirements.txt
```


1. Download DCASE2023T2 Dataset

```
bash download_dataset_2023.sh
```

2. Do patch for timm

- Modify timm_path to your timm_path
```
bash timm_patch.sh
```

3. Rename the evaluation Dataset
[GT attributes](https://github.com/nttcslab/dcase2023_task2_evaluator/tree/main/ground_truth_attributes)

- Go to the above link and download csv files for test datasets.
- For each toy in test folder, move 'rename_test_2023.py' file there and run it. Before running, please check 'folder_path' and 'csv_file_path' in python file.

4. For AudioMAE pretrained encoder, please go to the link below, download the pretrained ckpt (pretrained.pth), and move it into ./checkpoint folder. (Or, you can change 'finetune' parameter in models.py AudioMAE_pretrained class.)