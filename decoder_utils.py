import torch
import torch.nn as nn
from data import generate_data
from encoder import GraphAttentionEncoder

class Env():
    def __init__(self,device, x, embed_dim=128):
        super().__init__()
        """ 
            x(batch,max_stacks,max_tiers)
            
            node_embeddings: (batch, n_stacks, embed_dim)
            
            
            
        """
        self.device = device
        self.x=x

        self.batch,self.max_stacks,self.max_tiers=x.size()
        self.node_embeddings = None
        self.embed_dim = embed_dim
        #target_stack(batch)在create_t1更新，表示每一时刻需要移动的是哪一列
        self.target_stack=None
        #True表示空了， False表示没空
        self.empty=torch.zeros([self.batch],dtype=torch.bool).to(self.device)

    def find_target_stack(self):
        #把每个stacks的最大值都拿出来,torch.max是拿出个对子{mx_val,mx_index}
        #mx_val(batch,stacks)
        mx_val=torch.max(self.x,dim=2)[0].to(self.device)
        #target_stack(batch)
        self.target_stack=torch.argmax(mx_val,dim=1).to(self.device)

    #更新empty
    def _update_empty(self):
        #bottom_val(batch,max_stacks)
        bottom_val=self.x[:,:,0].to(self.device)
        #batch_mx(batch)
        batch_mx=torch.max(bottom_val,dim=1)[0].to(self.device)
        self.empty=torch.where(batch_mx>0.,False,True).to(self.device)

    # 把那些已经在顶部的移除
    def clear(self):
        self.find_target_stack()
        #反正那些空的移动了还是空的，所以直接把最大值坐标搞出来，
        # 再把顶层值搞出来，然后再全变0，然后另一边再添加,ll和cost不用冲

        #len_mask(batch,max_stacks,max_tiers)表示把len的值都弄出来变成1和0
        len_mask=torch.where(self.x>0.,1,0).to(self.device)
        #stack_len(batch,max_stacks)表示每一列最高是多少
        stack_len=torch.sum(len_mask,dim=2).to(self.device)

        #self.target_stack[:,None](batch,1)
        #target_stack_len(batch,1)[i][0]=stack_len[i][target_stack[i][0]] 相当于把目标列的长度都拿出来了
        target_stack_len=torch.gather(stack_len,dim=1,index=self.target_stack[:,None].to(self.device))

        #stack_mx_index(batch,max_stacks) 把每一列的最大值的下标找出来
        stack_mx_index=torch.argmax(self.x,dim=2).to(self.device)
        #target_stack_mx_index(batch,1)[i][0]=stack_mx_index[i][target_stack[i][0]]把目标列的最大值的坐标拿出来
        target_stack_mx_index=torch.gather(stack_mx_index,dim=1,index=self.target_stack[:,None].to(self.device)).to(self.device)

        #clear_mask(batch,1)表示哪些数据的最大值是能被删除的
        clear_mask=((target_stack_len-1)==target_stack_mx_index)
        clear_mask=clear_mask.to(self.device)
        #对于那些最大值已经是0的也就是完全消除完成的数据组，我们也不用管他，标记为False表示我们不需要填数
        clear_mask=clear_mask & (torch.where(target_stack_len>0,True,False).to(self.device))

        while torch.sum(clear_mask.squeeze(-1))>0 :
            #先把batch_mask变成(batch*max_stacks*max_tiers)的一维标记数组再reshape
            batch_mask=clear_mask.repeat_interleave(self.max_stacks*self.max_tiers).to(self.device)
            batch_mask=torch.reshape(batch_mask,(self.batch,self.max_stacks,self.max_tiers)).to(self.device)

            mask=torch.zeros((self.batch,self.max_stacks,self.max_tiers),dtype=torch.bool).to(self.device)
            input_index=(torch.arange(self.batch).to(self.device),self.target_stack,target_stack_len.squeeze(-1).to(self.device)-1)
            mask=mask.index_put(input_index,torch.tensor(True).to(self.device)).to(self.device)
            #batch_mask表示这一组数据可以清除最大值
            mask=mask & batch_mask
            mask=mask.to(self.device)
            self.x=self.x.masked_fill((mask==True).to(self.device),0.)

            #repeat same operations
            self.find_target_stack()
            len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
            stack_len = torch.sum(len_mask, dim=2).to(self.device)
            target_stack_len = torch.gather(stack_len, dim=1, index=self.target_stack[:, None].to(self.device)).to(self.device)
            stack_mx_index = torch.argmax(self.x, dim=2).to(self.device)
            target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:, None].to(self.device)).to(self.device)
            clear_mask=((target_stack_len-1)==target_stack_mx_index)
            clear_mask = clear_mask.to(self.device)
            clear_mask=clear_mask & (torch.where(target_stack_len>0,True,False).to(self.device))

        #每次清空完之后更新empty数组
        self._update_empty()


    def _get_step(self, next_node):
        """ next_node : (batch, 1) int, range[0, max_stacks)

            mask(batch,max_stacks,1) 1表示那一列不可选，0表示可选
            context: (batch, 1, embed_dim)
        """

        # len_mask(batch,max_stacks,max_tiers)表示把len的值都弄出来变成1和0
        len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
        # stack_len(batch,max_stacks)表示每一列最高是多少
        stack_len = torch.sum(len_mask, dim=2)
        # self.target_stack[:,None](batch,1)
        # target_stack_len(batch,1)[i][0]=stack_len[i][target_stack[i][0]] 相当于把目标列的长度都拿出来了

        target_stack_len = torch.gather(stack_len, dim=1, index=self.target_stack[:, None]).to(self.device)

        #next_stack_len(batch,1)[i][0]=stack_len[i][next_node[i][0]]
        next_stack_len=torch.gather(stack_len,dim=1,index=next_node).to(self.device)

        #  top_val(batch,max_stacks,1)=x(batch,max_stacks,stack_len(i,j,1))
        #top_ind(batch,max_stacks)-1 ,再用where把空的边0
        top_ind=stack_len-1
        top_ind=torch.where(top_ind>=0,top_ind,0).to(self.device)
        #top_val(batch,max_stacks,1)=x(batch,max_stacks,top_ind(i,j,1))
        top_val=torch.gather(self.x,dim=2,index=top_ind[:,:,None]).to(self.device)
        #top_val(batch,max_stacks)
        top_val=top_val.squeeze(-1)
        #target_top_val(batch,1)=top_val(batch,self.target_stack[batch,1])
        target_top_val=torch.gather(top_val,dim=1,index=self.target_stack[:,None]).to(self.device)

        #target_ind(batch,1)
        target_ind=target_stack_len-1
        target_ind=torch.where(target_ind>=0,target_ind,0).to(self.device)
        input_index=(torch.arange(self.batch).to(self.device),self.target_stack.to(self.device),target_ind.squeeze(-1).to(self.device))
        self.x=self.x.index_put_(input_index,torch.Tensor([0.]).to(self.device))


        input_index=(torch.arange(self.batch).to(self.device),next_node.squeeze(-1).to(self.device),next_stack_len.squeeze(-1).to(self.device))
        self.x=self.x.index_put_(input_index,target_top_val.squeeze(-1)).to(self.device)

        self.clear()




    # 初始化调用返回第一步的mask_t1和step_context_t1，step_context我们就设成初始点的embedding就行
    def _create_t1(self):
        #先吧self.target_stack更新出来 #target_stack(batch) 当前要移动的列
        self.find_target_stack()
        #mask(batch,max_stacks,1) 1表示那一列不可选，0表示可选
        mask_t1 = self.create_mask_t1()
        #step_context = target_stack_embedding(batch, 1, embed_dim)
        step_context_t1= self.create_context_t1()
        return mask_t1, step_context_t1

    # 创建初始化的mask 哪些列不能用
    def create_mask_t1(self):

        top_val=self.x[:,:,-1]
        #如果最后一列是0，那么这列就是可选的
        mask=torch.where(top_val>0,True,False).to(self.device)
        mask = mask.bool()

        #当前target_stack也不能走,这几句相当于mask[i][target_stack[i]]=True
        a=self.target_stack.clone().to(self.device)
        index = (torch.arange(self.batch).to(self.device), a.squeeze())
        mask=mask.index_put(index,torch.BoolTensor([True] ).to(self.device))


        #mask(batch,max_stacks,1) 1表示那一列不可选，0表示可选
        return mask[:,:,None].to(self.device)

    # 创建初始化的context
    def create_context_t1(self):



        # node_embeddings(batch,max_stacks,embed_dim) 然后把target_stack变成(batch,1,1)后把最后一维循环embed_dim变成(batch,1,128) 然后使用gather dim=1 就相当于
        # target_stack_embedding(batch,1,embed_dim)=node_embeddings(i,idx[i][j][k],k)，就是把目标列的embeding全部拿出来了

        target_stack_embedding = torch.gather(input=self.node_embeddings, dim=1,
                                       index=self.target_stack[:, None, None].repeat(1, 1, self.embed_dim))
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4


        return target_stack_embedding

    def get_log_likelihood(self, _log_p, pi):
        """ _log_p: (batch, decode_step, n_nodes)
            pi: (batch, decode_step), predicted tour
        """
        # log_p(batch,decode_step,1) 相当于把pi这个路径的所有log_p值给弄出来了
        # 相当于log_p(i,j,k=0)=_log_p(i,j,pi[i][j][0])
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        # 先把最后1维的1给消掉，然后再加起来
        # 由于是log_softmax,所以概率本来要用乘法，取了log就可以加法了
        return torch.sum(log_p.squeeze(-1), 1)


    def all_empty(self):
        #self.empty(batch)
        sum=torch.sum(self.empty.type(torch.int))
        if (sum==self.batch) :
            return True
        else:
            return False

class Sampler(nn.Module):
    """ args; logits: (batch, n_nodes)
        return; next_node: (batch, 1)
        TopKSampler <=> greedy; sample one with biggest probability
        CategoricalSampler <=> sampling; randomly sample one from possible distribution based on probability
    """

    def __init__(self, n_samples=1, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples


class TopKSampler(Sampler):
    def forward(self, logits):
        return torch.topk(logits, self.n_samples, dim=1)[1]  # == torch.argmax(log_p, dim = 1).unsqueeze(-1)


class CategoricalSampler(Sampler):
    def forward(self, logits):
        return torch.multinomial(logits.exp(), self.n_samples)

if __name__ == '__main__':
    batch=2
    n_containers=8
    max_stacks=4
    max_tiers=4
    emded_dim=128
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset=generate_data(device=device,n_samples=batch,n_containers=n_containers,max_stacks=max_stacks,max_tiers=max_tiers)
    node_embeddings=torch.rand( (batch,max_stacks,emded_dim) ,device=device)

    env = Env(dataset, node_embeddings)

    x=env.x
    env.clear()
    x=env.x


    print("debug in decoder_utils.py")

