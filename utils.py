import os
import glob
import joblib
import re
import csv
import itertools
import numpy as np
import torch
import pandas as pd

def path_to_dict(args, db_path, machine_type, dataset, section=None, domain=None):
    path_list = []
    if dataset == "train":
        wav_path = os.path.join(args.dataset_dir, machine_type, dataset)
        files = create_file_list(os.path.split(wav_path)[0], section='section_00', dir_name=dataset)
        path_list.extend(files)
        print(f'{machine_type} section 00 were split to {len(files)} wav files.')

    elif dataset == "eval":
        wav_path = os.path.join(args.dataset_dir, machine_type, 'test')
        files = create_file_list(os.path.split(wav_path)[0], section='section_00', dir_name='test')
        path_list.extend(files)
        print(f'{machine_type} section 00 {domain} were split to {len(files)} wav files!')

    elif dataset == "test":
        wav_path = os.path.join(args.test_dir, machine_type, 'test')
        files = create_file_list(os.path.split(wav_path)[0], section='section_00', dir_name='test')
        path_list.extend(files)
        print(f'{machine_type} section 00 {domain} were split to {len(files)} wav files!')

    else:
        raise ValueError("'dataset' must be one of 'train', 'eval', or 'test'.")
    
    os.makedirs(args.pre_data_dir, exist_ok=True)
    with open(db_path, 'wb') as f:
        joblib.dump(path_list, f)

# Train이면 label이 0이고, Test면 label이 1인 상태
# Check: Normal이면 label이 0이고, anomaly면 label이 1인 상태가 맞지 않나? 이렇게 바꾸자.
# 일단 utils.create_file_list는 바로위에서만 부르고, label이 필요가 없으니 그닥 중요하진 않은 듯 하다.
# 나중에 그냥 아예 없애는 것도 좋겠다.

def create_file_list(target_dir, section, dir_name, prefix_normal='normal', prefix_anomaly='anomaly', ext='wav'):
    section_name = section

    source_train_normal_files_path = f'{target_dir}/{dir_name}/{section_name}_source_train_{prefix_normal}*.{ext}'
    source_train_normal_files = sorted(glob.glob(source_train_normal_files_path))

    target_train_normal_files_path = f'{target_dir}/{dir_name}/{section_name}_target_train_{prefix_normal}*.{ext}'
    target_train_normal_files = sorted(glob.glob(target_train_normal_files_path))

    source_test_anomaly_files_path = f'{target_dir}/{dir_name}/{section_name}_source_test_{prefix_anomaly}*.{ext}'
    source_test_anomaly_files = sorted(glob.glob(source_test_anomaly_files_path))

    source_test_normal_files_path = f'{target_dir}/{dir_name}/{section_name}_source_test_{prefix_normal}*.{ext}'
    source_test_normal_files = sorted(glob.glob(source_test_normal_files_path))

    target_test_anomaly_files_path = f'{target_dir}/{dir_name}/{section_name}_target_test_{prefix_anomaly}*.{ext}'
    target_test_anomaly_files = sorted(glob.glob(target_test_anomaly_files_path))

    target_test_normal_files_path = f'{target_dir}/{dir_name}/{section_name}_target_test_{prefix_normal}*.{ext}'
    target_test_normal_files = sorted(glob.glob(target_test_normal_files_path))

    files = np.concatenate((source_train_normal_files, target_train_normal_files, source_test_anomaly_files, source_test_normal_files,
                            target_test_anomaly_files, target_test_normal_files), axis=0)

    return files

def select_dirs(args):
    dir_path = os.path.abspath(f'{args.dataset_dir}/*')
    dirs = sorted(glob.glob(dir_path))
    output_dirs = [v for v in dirs if os.path.split(v)[1] in args.machines]
    return output_dirs

def get_domain_type(file_name):
    if 'source' in file_name:
        return 0
    elif 'target' in file_name:
        return 1
    return -1
    
def file_name_to_label(args, file_name):
    n_class = args.nclass

    one_hot = torch.zeros(n_class)

    class_num = get_class_num(args, file_name) 
    one_hot[class_num] = 1

    return one_hot

def get_class_num(args, file_name):
    
    df = pd.read_csv(args.class_info_file)
    for i, feature in enumerate(df['feature']):
        if feature in file_name:
            return i
            break

    raise ValueError("Not implemented Correctly")

def save_checkpoint(args, model, epoch, path, losstracker):
    save_model_path = f'{args.model_dir}/{args.version}/{path}'
    state = {'epoch': epoch,
             'model_state_dict': model.module.state_dict() if args.distributed else model.state_dict()}
    torch.save(state, save_model_path)
    history_img = f'{os.path.split(save_model_path)[0]}/history.png'
    losstracker.save_figure(history_img)

def create_test_label(path_db):
    file_list = joblib.load(path_db)
    y_true = np.array([0 if ('normal' in v) else 1 for v in file_list])
    return y_true, file_list

def save_csv(save_file_path, save_data):
    with open(save_file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)