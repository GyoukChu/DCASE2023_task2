import numpy as np

import torch.nn as nn
import torch.optim as optim
from CosineWarmup import CosineAnnealingWarmupRestarts
import torch
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc = nn.Linear(10,5)
    def forward(self, x):
        return self.fc(x) 

def visualize_learning_rate(scheduler, epochs):
    lrs = [[] for _ in range(len(scheduler.optimizer.param_groups))]
    for epoch in range(epochs):
        for lst, dct in zip(lrs, scheduler.optimizer.param_groups):
            lst.append(dct['lr'])
#         lrs.append(scheduler.optimizer.param_groups[0]["lr"])
        scheduler.step()
    
    lists = []
    for l in lrs:
        lists.append(list(range(epochs)))
        lists.append(l)
    plt.figure(figsize=(12,4)) 
    lines = plt.plot(*lists)
    plt.setp(lines[0], linewidth=3)
    plt.title('Learning rate change')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.xticks(np.arange(0, epochs + 1, 10), rotation=45)
    plt.yticks(np.arange(1e-4, 1.1e-3, 1e-4))
    plt.grid(True)
    plt.savefig("test.png")

model = TestNet()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=100, max_lr=1e-3, min_lr=1e-6, warmup_steps=4)
visualize_learning_rate(scheduler, epochs=500)