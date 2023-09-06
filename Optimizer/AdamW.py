import torch

def Optimizer(parameters, lr, betas=(0.9, 0.999), weight_decay=0.01):
    
    optim_fn=torch.optim.AdamW(parameters, lr = lr, betas=betas, weight_decay = weight_decay)
    
    print('Initialised AdamW optimizer')
    
    return optim_fn