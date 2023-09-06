import torch
import torch.nn as nn
from tqdm import tqdm

import pprint
import os
import time

from torchmetrics import MeanMetric
from sklearn import metrics

import utils
import dataset

class ModelAlreadyFoundException(Exception):
    def __init__(self, value):
        self.value = value   
    def __str__(self):
        return self.value

# TODO: Multi-GPU시 rank=0일 때만 print 되게 설정
class Trainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        # TODO: Multi-GPU 시 모델
        self.model = kwargs['model'].to(self.args.gpu)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.losstracker = kwargs['losstracker']

    def loss_function(self, label, pred):
        l1 = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss = l1(pred, label)

        return loss

    def train(self, train_dataloader):
        try:
            os.makedirs(os.path.join(self.args.model_dir, self.args.version))
        except:
            raise ModelAlreadyFoundException('Such version of model exists already. Please change the version.')

        best_epoch = 0
        best_score = 0 # harmonic mean
        best_results = dict() # Check: (machine_type, domain)이 key, (AUC, pAUC)가 value
        
        for epoch in range(self.args.epochs):
            
            metric = MeanMetric().to(self.args.gpu)
            time.sleep(0.01)

            # https://discuss.pytorch.org/t/why-is-sampler-set-epoch-epoch-needed-for-distributedsampler/149672
            if self.args.distributed:
                train_dataloader.sampler.set_epoch(epoch)

            pbar = tqdm(train_dataloader, total=len(train_dataloader), ncols=100)
            
            self.model.train()
            # TODO: Multi-GPU 시, data가 self.args.gpu에 올라가면 되는가.
            for mel_spec, label in pbar:

                mel_spec = mel_spec.to(self.args.gpu)
                label = label.to(self.args.gpu)

                pred = self.model(mel_spec)

                loss = self.loss_function(label, pred)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                metric.update(loss)
                if self.args.gpu == 0: # Check: Multi-GPU 시에도 한번만 출력
                    pbar.set_description(f'Epoch: {epoch}\tLoss: {metric.compute():.4f}')
                
            if epoch % self.args.test_epochs == 0:
                self.model.eval()
                score, results = self.validate()
                if score > best_score:
                    checkpoint_path = 'checkpoint_best_model_pth.tar'
                    utils.save_checkpoint(args=self.args,
                                        model=self.model,
                                        epoch=epoch,
                                        path=checkpoint_path,
                                        losstracker=self.losstracker)
                    print(f'Model saved! \t New Best Score: {score}')
                    best_epoch, best_score, best_results = epoch, score, results
                    pprint.pprint(results)
            
            self.scheduler.step()
            self.losstracker.add_train_loss(metric.compute().cpu())
            
        print(f'Training completed. Best Epoch: {best_epoch:d}\t Best Score(in validation): {best_score:2.4f}')
        return
    

    def validate(self):
        results = dict() # (machine_type, domain)이 key, (AUC, pAUC)가 value
        
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)

        for domain in ['source', 'target']:
            machine_type_list = machine_type_list = ['ToyCar','ToyTrain','fan','gearbox','bearing','slider','valve']
                         # 'Vacuum','ToyTank','ToyNscale','ToyDrone','bandsaw','grinder','shaker'] # TODO: cross validation HERE
            for machine_type in machine_type_list:
                y_pred = []
                path_db = os.path.join(self.args.pre_data_dir, f'eval_data_2023_{machine_type}_{domain}.db')
                eval_dataloader = dataset.generate_val_test_dataloader(self.args, machine_type, path_db)
                y_true, file_list = utils.create_test_label(path_db)
                        
                for mel_spec, label in eval_dataloader:
                    mel_spec = mel_spec.to(self.args.gpu)
                    label = label.to(self.args.gpu)
                    with torch.no_grad():
                        pred = self.model(mel_spec)

                    score = self.anomaly_score(label, pred)
                    y_pred.extend(score.detach().cpu().numpy())

                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                results[(machine_type, domain)] = (auc, p_auc)
        
        entire_list = list(results.values())
        entire_list = [item for sublist in entire_list for item in sublist] # list of tuples -> list of floats
        score = len(entire_list) / sum([1 / ele for ele in entire_list])
        return score, results

    def anomaly_score(self, label, pred):
        pred_prob = nn.functional.softmax(pred, dim=-1)
        loss = torch.sum(label * pred_prob, dim=1)
        return 1 - loss
    
    def test(self):
        self.model.eval()
        results = []
        csv_lines = []
        csv_lines.append(['Final_Score'])
        csv_lines.append([])

        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)

        for domain in ['source', 'target']:
            machine_type_list = ['Vacuum','ToyTank','ToyNscale','ToyDrone','bandsaw','grinder','shaker']
            for machine_type in machine_type_list:
                csv_lines.append([domain])
                csv_lines.append(['machine_type', 'AUC', 'pAUC'])
                y_pred = []
                anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{machine_type}_{domain}.csv'
                anomaly_score_list = [['Name', 'Anomaly Score']]
                
                eval_test = 'test'
                path_db = os.path.join(self.args.pre_data_dir, f'{eval_test}_data_2023_{machine_type}_{domain}.db')
                test_dataloader = dataset.generate_val_test_dataloader(self.args, machine_type, path_db)
                y_true, file_list = utils.create_test_label(path_db)

                for mel_spec, label in test_dataloader:
                    mel_spec = mel_spec.to(self.args.gpu)
                    label = label.to(self.args.gpu)
                    with torch.no_grad():
                        pred = self.model(mel_spec)

                    score = self.anomaly_score(label, pred)
                    y_pred.extend(score.detach().cpu().numpy())
                        
                anomaly_score_list.extend([os.path.split(a)[1], b] for a, b in zip(file_list, y_pred))
                
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
                results.extend([auc, p_auc])
                print(f'{machine_type} section 00 {domain} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
                utils.save_csv(anomaly_score_csv, anomaly_score_list)
                csv_lines.append([machine_type, auc, p_auc])

        score = len(results) / sum([1 / ele for ele in results])
        score_percentange = score * 100
        print(f'Total Score: {score_percentange:3.3f}%')
        
        result_path = os.path.join(os.path.join(self.args.result_dir, self.args.version), f'_{self.args.result_file}')
        utils.save_csv(result_path, csv_lines)
        
        return