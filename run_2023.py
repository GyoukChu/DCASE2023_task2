import os
import numpy as np
import torch
import random
import yaml
import argparse

import dataset
import models
import utils
import losstracker
from trainer import Trainer

from Optimizer.AdamW import Optimizer
from Scheduler.CosineWarmup import CosineAnnealingWarmupRestarts

# TODO: Multi-GPU
# TODO: Multi-GPU시 rank=0일 때만 print 되게 설정

def set_random_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def yaml_load():
    with open("param_2023.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

def get_argparse(param):
    parser = argparse.ArgumentParser()
    # Directory Path
    parser.add_argument('--dataset-dir', default=param['dataset_dir'], type=str, help='dataset dir')
    parser.add_argument('--test-dir', default=param['test_dir'], type=str, help='evaluation dataset dir')
    parser.add_argument('--pre-data-dir', default=param['pre_data_dir'], type=str, help='preprocess data dir')
    parser.add_argument('--model-dir', default=param['model_dir'], type=str, help='model dir')
    parser.add_argument('--result-dir', default=param['result_dir'], type=str, help='result dir')
    parser.add_argument('--result-file', default=param['result_file'], type=str, help='result file name')
    parser.add_argument('--machines', default=param['machines'], nargs='+', type=str, help='allowed processing machine')
    # Model
    parser.add_argument('--version', default=param['version'], type=str, help='version')
    parser.add_argument('--class-info-file', default=param['class_info_file'], type=str, help='class info file name')
    parser.add_argument('--nclass', default=param['nclass'], type=int, help='number of classes for classifier')
    # Spectrogram
    parser.add_argument('--sample-rate', default=param['sample_rate'], type=int, help='STFT sampling rate')
    parser.add_argument('--n-fft', default=param['n_fft'], type=int, help='STFT n_fft')
    parser.add_argument('--win-length', default=param['win_length'], type=int, help='STFT win length')
    parser.add_argument('--hop-length', default=param['hop_length'], type=int, help='STFT hop length')
    parser.add_argument('--n-mels', default=param['n_mels'], type=int, help='STFT n_mels')
    parser.add_argument('--frames', default=param['frames'], type=int, help='STFT time frames')
    parser.add_argument('--power', default=param['power'], type=float, help='STFT power')
    # Training
    parser.add_argument('--batch-size', default=param['batch_size'], type=int, help='batch size')
    parser.add_argument('--epochs', default=param['epochs'], type=int, help='training epochs')
    parser.add_argument('--test-epochs', default=param['test_epochs'], type=int, help='inference every n epochs')
    parser.add_argument('--lr', default=param['lr'], type=float, help='initial learning rate')
    parser.add_argument('--num-workers', default=param['num_workers'], type=int, help='number of workers for dataloader')
    parser.add_argument('--device-ids', default=param['device_ids'], nargs='+', type=int, help='gpu ids')
    # Others
    parser.add_argument('--max-fpr', default=param['max_fpr'], type=float, help='max fpr for pAUC')
    return parser

def preprocess(args):
    dirs = utils.select_dirs(args)
    for path in dirs:
        dataset.generate_dataset(args, path, dataset='train')
        dataset.generate_dataset(args, path, dataset='eval')
        dataset.generate_dataset(args, path, dataset='test')

def train(args):
    
    machine_type_list = ['ToyCar','ToyTrain','fan','gearbox','bearing','slider','valve',
                         'Vacuum','ToyTank','ToyNscale','ToyDrone','bandsaw','grinder','shaker']
    
    model = models.AudioMAE_pretrained(n_class=args.nclass)

    train_losstracker = losstracker.LossTracker()

    train_dataloader = dataset.generate_train_dataloader(args, machine_type_list)

    optimizer = Optimizer(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=100, max_lr=1e-3, min_lr=1e-6, warmup_steps=4)

    trainer = Trainer(args=args,
                      model=model,
                      losstracker=train_losstracker,
                      optimizer=optimizer,
                      scheduler=scheduler)

    trainer.train(train_dataloader)
    return

def test(args):
    model_path = f'{args.model_dir}/{args.version}/checkpoint_best_model_pth.tar'
    model = models.AudioMAE_pretrained(n_class=args.nclass) 
    model.load_state_dict(torch.load(model_path)['model_state_dict'])

    trainer = Trainer(args=args,
                      model=model,
                      losstracker=None,
                      optimizer=None,
                      scheduler=None)
    
    trainer.test()
    return

def main():
    set_random_everything(42)

    param = yaml_load()
    parser = get_argparse(param)
    args = parser.parse_args()
    # read parameters from command line
    args = parser.parse_args(namespace=args)

    preprocess(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(v) for v in args.device_ids])
    # TODO: Multi-GPU (일단 적어도 DP라도 되게.)
    if (not torch.cuda.is_available()) or (torch.cuda.device_count()!=1):
        raise GPUException('Only training with exactly 1 gpu is available now.')
    args.gpu = 0
    args.distributed = False
    
    train(args)
    test(args)

class GPUException(Exception):
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return self.value

if __name__ == "__main__":
    main()