import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import sys
from tqdm import tqdm
import joblib

import utils
from DistributedEvalSampler import DistributedEvalSampler

def file_load(file_name):
    try:
        y, sr = torchaudio.load(file_name)
        return y, sr
    except:
        print("file_broken or not exists!! : {}".format(file_name))

def file_to_log_mel_spectrogram(args, file_name):
    y, sr = file_load(file_name)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate,
                                                     n_fft=args.n_fft,
                                                     win_length=args.win_length,
                                                     hop_length=args.hop_length,
                                                     n_mels=args.n_mels,
                                                     power=args.power,
                                                     # normalized=True,
                                                     pad_mode='constant', norm='slaney', mel_scale='slaney')
    mel_spectrogram = transform(y)
    log_mel_spectrogram = 20.0 / args.power * torch.log10(mel_spectrogram + sys.float_info.epsilon)

    n_frames = log_mel_spectrogram.shape[2]
    p = args.frames - n_frames

    if p<0:
        log_mel_spectrogram = log_mel_spectrogram[:,:,0:args.frames]
    elif p==0:
        pass
    else:
        m = torch.nn.ZeroPad2d((0,p,0,0))
        log_mel_spectrogram = m(log_mel_spectrogram)

    return log_mel_spectrogram

# No label
def list_to_dataset(args, file_list):
    num_data = len(file_list)
    dataset_array = np.zeros((num_data, args.n_mels, args.frames))
    for ii in tqdm(range(num_data), desc='generate dataset'):
        log_mel = file_to_log_mel_spectrogram(args, file_list[ii])
        dataset_array[ii] = log_mel
    return dataset_array

def save_dataset_mean_std(args, data_db, mean_path, std_path):
    file_list = joblib.load(data_db)
    dataset = list_to_dataset(args, file_list)
    dataset_mean, dataset_std = np.mean(dataset, axis=(0, 2)), np.std(dataset, axis=(0, 2))
    np.save(mean_path, dataset_mean)
    np.save(std_path, dataset_std)

def generate_dataset(args, path, dataset):
    machine_type = os.path.split(path)[1] # Check: machine type 제대로 들어가는지 확인

    if dataset == "train":
        data_db = os.path.join(args.pre_data_dir, f'train_data_2023_{machine_type}.db')
        if not os.path.exists(data_db):
            utils.path_to_dict(args, data_db, machine_type, dataset)
        mean_path = f'{args.pre_data_dir}/train_dataset_2023_{machine_type}_mean.npy'
        std_path = f'{args.pre_data_dir}/train_dataset_2023_{machine_type}_std.npy'
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            save_dataset_mean_std(args, data_db, mean_path, std_path)

    elif dataset == "eval":
        for domain in ['source', 'target']:
            data_db = os.path.join(args.pre_data_dir, f'eval_data_2023_{machine_type}_{domain}.db')
            if not os.path.exists(data_db):
                utils.path_to_dict(args, data_db, machine_type, dataset, domain=domain)

    else:
        for domain in ['source', 'target']:
            data_db = os.path.join(args.pre_data_dir, f'test_data_2023_{machine_type}_{domain}.db')
            if not os.path.exists(data_db):
                utils.path_to_dict(args, data_db, machine_type, dataset, domain=domain)

class Generator(torch.nn.Module):
    def __init__(self, args, machine_type):
        super(Generator, self).__init__()
        self.args = args
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=args.sample_rate,
                                                     n_fft=args.n_fft,
                                                     win_length=args.win_length,
                                                     hop_length=args.hop_length,
                                                     n_mels=args.n_mels,
                                                     power=args.power,
                                                     # normalized=True, - time normalization X
                                                     pad_mode='constant', norm='slaney', mel_scale='slaney')
        self.train_mean = torch.FloatTensor(np.load(os.path.join(args.pre_data_dir, f'train_dataset_2023_{machine_type}_mean.npy')))
        self.train_std = torch.FloatTensor(np.load(os.path.join(args.pre_data_dir, f'train_dataset_2023_{machine_type}_std.npy')))

    def __call__(self, x):
        y, _ = file_load(x)
        
        mel_spectrogram = self.transform(y)
        log_mel_spectrogram = 20.0 / self.args.power * torch.log10(mel_spectrogram + sys.float_info.epsilon)

        n_frames = log_mel_spectrogram.shape[2]
        p = self.args.frames - n_frames

        if p<0:
            log_mel_spectrogram = log_mel_spectrogram[:,:,0:self.args.frames]
        elif p==0:
            pass
        else:
            m = torch.nn.ZeroPad2d((0,p,0,0))
            log_mel_spectrogram = m(log_mel_spectrogram)

        norm_log_mel = self.frequency_normalize(log_mel_spectrogram, self.train_mean, self.train_std)
        return norm_log_mel

    def frequency_normalize(self, log_mel, mean, std):
        return torch.unsqueeze(torch.transpose((torch.transpose(torch.squeeze(log_mel), 0, 1) - mean) / std, 0, 1), dim=0)
    
class DCASEDataset(Dataset):
    def __init__(self, args, machine_type, path_db):
        super(DCASEDataset, self).__init__()
        self.args = args
        self.machine_type = machine_type
        self.file_list = joblib.load(path_db)
        self.generator = Generator(args, machine_type)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        log_mel = self.generator(file_name)
        domain_type = utils.get_domain_type(file_name)
        label = utils.file_name_to_label(self.args, file_name)

        return log_mel, label

    def __len__(self):
        return len(self.file_list)

def generate_train_dataloader(args, machine_type_list):
    
    machine_type = machine_type_list[0]
    path_db = os.path.join(args.pre_data_dir, f'train_data_2023_{machine_type}.db')
    dataset = DCASEDataset(args, machine_type, path_db)

    for index, machine_type in enumerate(machine_type_list):
        if index==0: continue
        else:
            path_db = os.path.join(args.pre_data_dir, f'train_data_2023_{machine_type}.db')
            dataset2 = DCASEDataset(args, machine_type, path_db)
            dataset = dataset + dataset2

    sampler = DistributedSampler(dataset, shuffle=True) if args.distributed else None

    # TODO: num_workers DDP 쓸 때 gpu 개수로 나눠주기
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None),
                                  num_workers=args.num_workers, pin_memory=True, sampler=sampler)
    
    return train_dataloader

# Check: test_dataloader는 DistributedEvalSampler 써야됨. Multi-GPU training 참고!

def generate_val_test_dataloader(args, machine_type, path_db):
    
    dataset = DCASEDataset(args, machine_type, path_db)
    
    sampler = DistributedEvalSampler(dataset, shuffle=False) if args.distributed else None

    # TODO: num_workers DDP 쓸 때 gpu 개수로 나눠주기
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, sampler=sampler)
    
    return dataloader
