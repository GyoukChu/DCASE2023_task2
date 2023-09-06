import torch

def Scheduler(optimizer, milestones, gamma=0.5):

	sche_fn = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
	
	print('Initialised MultiStepLR scheduler')

	return sche_fn