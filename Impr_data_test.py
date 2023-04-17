from time import time
import torch
import random
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

def permute(nums):
	def permutation(nums,k,n):
		if k==n:
			return res.append(nums[:])
		for i in range(k,n):
			nums[i],nums[k]=nums[k],nums[i]
			permutation(nums,k+1,n)
			nums[i],nums[k]=nums[k],nums[i]
	res=[]
	permutation(nums,0,len(nums))
	return res

def choose_num(per,l,num):
	if num>l :
		raise ValueError("can not sample so many permutation")

	id=[i for i in range(l)]
	sample_id=random.sample(id,num)
	res = []
	for i in range(len(sample_id)):
		res.append(per[sample_id[i]])
	return res

if __name__ == '__main__':
	args = test_parser()
	t1 = time()
	device = torch.device('cuda:2')
	torch.cuda.set_device(device)
	pretrained = load_model(device='cuda:2',path=args.path, embed_dim=128, n_containers=args.n_containers, max_stacks=args.max_stacks
							,max_tiers=args.max_tiers,n_encode_layers=3)
	pretrained=pretrained.to(device)
	out_path=args.out_path
	#out_path=None

	print(f'model loading time:{time() - t1}s')
	if args.txt is not None:
		data=data_from_txt(args.txt)
	else:
		data = generate_data(device,args.batch,args.n_containers,args.max_stacks,args.max_tiers)


	num=args.impr_num  # number of permutations
	a=[i for i in range(pretrained.max_stacks)]
	per=permute(a)
	num=min(num,len(per))

	per=choose_num(per,len(per),num)
	inst_num,s,t = data.shape


	impr_data=torch.zeros((len(per),inst_num,s,t))
	for i in range(len(per)):
		tmp_data=torch.zeros((inst_num,s,t))
		for j in range(inst_num):
			now_inst=data[j]
			tmp_inst=torch.zeros((s,t))
			for k in range(s):
				tmp_inst[k]=(now_inst[per[i][k]])
			tmp_data[j]=tmp_inst
		impr_data[i]=(tmp_data)

	#data (inst_num,s,t)
	data=data.to(device)
	# impr_data (len(per),inst_num,s,t)
	impr_data=impr_data.to(device)


	# greedy rollout once  sample_num=0
	t1=time()
	sample_num = 0  # limit of num of sample
	sample_num_limit = args.sampl_num
	pretrained.eval()
	with torch.no_grad():
		mi = rollout(model=pretrained, dataset=data)
	print('mi.mean():', mi.mean())

	# begin to count sample_num
	for i in range(len(per)):

		if sample_num >= sample_num_limit :
			break

		pretrained.eval()
		with torch.no_grad():
			L = rollout(model=pretrained, dataset=impr_data[i])
		mi = torch.minimum(mi, L).to(device)
		sample_num+=1

	#sampling
	while True :

		if sample_num >= sample_num_limit :
			break

		for i in range(len(per)):

			if sample_num >= sample_num_limit:
				break

			pretrained.eval()
			with torch.no_grad():
				#impr_data[i] (inst_num,s,t)
				L, ll = pretrained(impr_data[i], decode_type='sampling')
			mi = torch.minimum(mi, L).to(device)
			sample_num+=1

	t2=time()

	if out_path!=None :
		with open(out_path,"w") as f:
			f.write('\n');
		for i in range(len(mi)):
			with open(out_path,"a") as f:
				f.write('%d\n'%mi[i]);

	print( 'after sample: %dmin%dsec' %((t2 - t1) // 60, (t2 - t1) % 60))
	#print(mi)
	print('mi.mean():',mi.mean())

	print('sample num: ',sample_num)