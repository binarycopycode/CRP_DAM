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

def find_target_container(layout):
	max_num=0.
	target_x=-1
	target_y=-1
	s,t=layout.shape
	for i in range(s):
		for j in range(t):
			if(layout[i][j]>max_num):
				max_num=layout[i][j]
				target_x=i
				target_y=j

	return target_x,target_y

def clear_top(device,layout):
	while True :
		x,y=find_target_container(layout)
		if x<0 :
			return layout
		# layout_mask(max_stacks,max_tiers)
		layout_mask = torch.where(layout > 0., 1, 0).to(device)
		# h_len (max_stacks)
		h_len = torch.sum(layout_mask, dim=1)
		if h_len[x]-1==y :
			layout[x][y] = 0.
		else :
			return layout

#layout (s,t)
def beam_search(device,layout,beam_size,model):

	#lst=[(torch.FloatTensor([[2,2],[2,2]]),2),(torch.FloatTensor([[3,3],[3,1]]),3),(torch.FloatTensor([[1,1],[1,1]]),1)]
	#lst.sort(key=lambda x:x[1])

	max_stacks,max_tiers=layout.shape
	ans=max_stacks*max_tiers
	now_layout=clear_top(device,layout)
	lst=[now_layout]
	step_count=0
	while len(lst)>0 :
		next_lst=[]
		step_count+=1

		for i in range(len(lst)):

			x,y=find_target_container(lst[i])
			if (x<0) :
				return ans
			# layout_mask(max_stacks,max_tiers)
			layout_mask = torch.where(lst[i] > 0., 1, 0).to(device)
			# h_len (max_stacks)
			h_len = torch.sum(layout_mask, dim=1)
			for j in range(max_stacks):
				if j==x or h_len[j]==max_tiers:
					continue
				new_layout=lst[i].clone()
				new_layout[j][h_len[j]]=lst[i][x][h_len[x]-1]
				new_layout[x][h_len[x]-1]=0.
				new_layout=clear_top(device,new_layout)
				new_data=torch.zeros((1,s,t))
				new_data[0]=new_layout
				new_data=new_data.to(device)

				model.eval()
				with torch.no_grad():
					mi = rollout(model=model, dataset=new_data)
				ans=min(ans,step_count+mi[0])
				next_lst.append( (new_layout,int(mi[0])) )


		next_lst.sort(key=lambda x:x[1])
		lst_len=min(len(next_lst),beam_size)
		lst=[]
		for i in range(lst_len):
			lst.append(next_lst[i][0])

	return ans

if __name__ == '__main__':
	#print("233")
	args = test_parser()
	t1 = time()
	device = torch.device('cuda:2')
	torch.cuda.set_device(device)
	pretrained = load_model(device='cuda:2',path=args.path, embed_dim=128, n_containers=args.n_containers, max_stacks=args.max_stacks
							,max_tiers=args.max_tiers,n_encode_layers=3)
	pretrained=pretrained.to(device)
	out_path=args.out_path

	print(f'model loading time:{time() - t1}s')

	if args.txt is not None:
		data = data_from_txt(args.txt)
	else:
		raise AssertionError

	inst_num,s,t = data.shape
	ans=torch.zeros(inst_num)


	for i in range(inst_num):
		ans[i]=beam_search(device=device,layout=data[i],beam_size=args.beam_size,model=pretrained)

	t2=time()

	if out_path != None:
		with open(out_path, "w") as f:
			f.write('\n');
		with open(out_path, "a") as f:
			f.write('beam_size:%d\n' % args.beam_size);
		for i in range(len(ans)):
			with open(out_path, "a") as f:
				f.write('%d\n' % ans[i]);
		with open(out_path, "a") as f:
			f.write('beam search time: %dmin%dsec' % ((t2 - t1) // 60, (t2 - t1) % 60));

	print('beam search time: %dmin%dsec' % ((t2 - t1) // 60, (t2 - t1) % 60))
	print('beam_search: ans.mean():', ans.mean())
