import torch

def Scheduler(optimizer, restart_interval, mult_factor=1, min_lr=0):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=restart_interval, T_mult=mult_factor, eta_min=min_lr)

	print('Initialised SGDR scheduler')

	return sche_fn