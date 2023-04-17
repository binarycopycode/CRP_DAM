from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import AttentionModel
from data import generate_data, data_from_txt,Generator

from baseline import load_model
from config import test_parser


def rollout(model, dataset, batch=1000, disable_tqdm=False):
	costs_list = []
	dataloader = DataLoader(dataset, batch_size=batch)
	# for inputs in tqdm(dataloader, disable=disable_tqdm, desc='Rollout greedy execution'):
	for t, inputs in enumerate(dataloader):
		with torch.no_grad():
			# ~ inputs = list(map(lambda x: x.to(self.device), inputs))
			cost, _ = model(inputs, decode_type='greedy')
			# costs_list.append(cost.data.cpu())
			costs_list.append(cost)
	return torch.cat(costs_list, 0)

if __name__ == '__main__':
	args = test_parser()
	t1 = time()
	device = torch.device('cuda:0')
	torch.cuda.set_device(device)
	pretrained = load_model(device='cuda:0',path=args.path, embed_dim=128, n_containers=args.n_containers, max_stacks=args.max_stacks
							,max_tiers=args.max_tiers,n_encode_layers=3)
	pretrained=pretrained.to(device)


	print(f'model loading time:{time() - t1}s')
	if args.txt is not None:
		data=data_from_txt(args.txt)
	else:
		data = generate_data(device,args.batch,args.n_containers,args.max_stacks,args.max_tiers)

	data=data.to(device)

	t1=time()
	sample_num=0  #limit of num of sample
	sample_num_limit = args.sampl_num
	pretrained.eval()
	with torch.no_grad():
		mi = rollout(model=pretrained, dataset=data)
	print('mi.mean():', mi.mean())

	while True:

		if sample_num == sample_num_limit :
			break

		pretrained.eval()
		with torch.no_grad():
			L, ll = pretrained(data, decode_type='sampling')
		mi=torch.minimum(mi,L).to(device)
		sample_num+=1


	t2=time()
	print( 'after sample: %dmin%dsec' %((t2 - t1) // 60, (t2 - t1) % 60))
	#print(mi)
	print('mi.mean():',mi.mean())
	print('sample num: ',sample_num)