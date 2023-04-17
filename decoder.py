import torch
import torch.nn as nn


from encoder import GraphAttentionEncoder
from layers import MultiHeadAttention, DotProductAttention
from data import generate_data
from decoder_utils import TopKSampler, CategoricalSampler, Env



class DecoderCell(nn.Module):
    def __init__(self,device , embed_dim=128, n_encode_layers=3, n_heads=8, clip=10.,FF_hidden=512,n_containers=8, max_stacks=4,max_tiers=4, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.embed_dim=embed_dim
        self.Encoder = GraphAttentionEncoder(embed_dim, n_heads, n_encode_layers, FF_hidden, n_containers, max_stacks,
                                             max_tiers)

        self.Wk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_step = nn.Linear(embed_dim , embed_dim, bias=False)

        self.MHA = MultiHeadAttention(n_heads=n_heads, embed_dim=embed_dim, need_W=False)
        self.SHA = DotProductAttention(clip=clip, return_logits=True, head_depth=embed_dim)
        # SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
        self.env = Env

    def compute_static(self, node_embeddings, graph_embedding):
        self.Q_fixed = self.Wq_fixed(graph_embedding[:, None, :])
        self.K1 = self.Wk1(node_embeddings)
        self.V = self.Wv(node_embeddings)
        self.K2 = self.Wk2(node_embeddings)

    def compute_dynamic(self, mask, step_context):
        Q_step = self.Wq_step(step_context)
        Q1 = self.Q_fixed + Q_step
        Q2 = self.MHA([Q1, self.K1, self.V], mask=mask)
        Q2 = self.Wout(Q2)
        logits = self.SHA([Q2, self.K2, None], mask=mask)
        return logits.squeeze(dim=1)

    # x是原始数据 x[0] -- depot_xy: (batch, 2) x[1] -- customer_xy: (batch, n_custmer=20, 2) x[2] -- demand: (batch, n_custmer=20)
    # return pi表示是否需要返回完整的路径
    def forward(self, x, n_containers=8, return_pi=False, decode_type='sampling'):


        env = Env(self.device,x,self.embed_dim)

        #先清理已经满足的
        env.clear()

        encoder_output=self.Encoder(env.x)
        # encoder_output 两项分别是(batch,max_stacks,embed_dim)和对这max_stacks求mean得到(batch,embed_dim)
        node_embeddings, graph_embedding = encoder_output
        env.node_embeddings=node_embeddings
        self.compute_static(node_embeddings, graph_embedding)

        #mask(batch,max_stacks,1) 1表示那一列不可选，0表示可选
        #step_context=target_stack_embedding(batch, 1, embed_dim) 表示要移动的目标列的embedding
        mask, step_context = env._create_t1()

        #default n_samples=1
        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
        #log_ps表示每次选的概率矩阵全存起来，tours表示每次选择的是哪个
        log_ps, tours = [], []
        #把3维大小拿出来
        batch,max_stacks,max_tiers = x.size()
        #代价
        cost=torch.zeros(batch).to(self.device)
        #每次选择的概率
        ll=torch.zeros(batch).to(self.device)

        for i in range(n_containers * max_tiers):

            # logits (batch,max_stacks)
            logits = self.compute_dynamic(mask, step_context)
            # log_p (batch,max_stacks)
            log_p = torch.log_softmax(logits, dim=-1)
            # next_node (batch,1) 表示当前
            next_node = selecter(log_p)

            #选出了next_node以后更新代价和
            cost += (1.0 - env.empty.type(torch.float64))
            #output(batch,1) (i,1)=logp[i][next_node[i][1]]
            #ll+=output(batch,1).squeeze(-1)
            ll += torch.gather(input=log_p,dim=1,index=next_node).squeeze(-1)

            #solv the actions
            env._get_step(next_node)

            if env.all_empty():
                break

            # re-compute node_embeddings
            encoder_output = self.Encoder(env.x)
            # encoder_output 两项分别是(batch,max_stacks,embed_dim)和对这max_stacks求mean得到(batch,embed_dim)
            node_embeddings, graph_embedding = encoder_output
            env.node_embeddings = node_embeddings
            self.compute_static(node_embeddings, graph_embedding)

            mask, step_context = env._create_t1()


        if return_pi:
            return cost, ll, pi
        return cost, ll


if __name__ == '__main__':
    batch, n_nodes, embed_dim = 5, 21, 128
    data = generate_data(n_samples=batch, n_customer=n_nodes - 1)
    decoder = DecoderCell(embed_dim, n_heads=8, clip=10.)
    node_embeddings = torch.rand((batch, n_nodes, embed_dim), dtype=torch.float)
    graph_embedding = torch.rand((batch, embed_dim), dtype=torch.float)
    encoder_output = (node_embeddings, graph_embedding)
    # a = graph_embedding[:,None,:].expand(batch, 7, embed_dim)
    # a = graph_embedding[:,None,:].repeat(1, 7, 1)
    # print(a.size())

    decoder.train()
    cost, ll, pi = decoder(data, encoder_output, return_pi=True, decode_type='sampling')
    print('\ncost: ', cost.size(), cost)
    print('\nll: ', ll.size(), ll)
    print('\npi: ', pi.size(), pi)

# cnt = 0
# for i, k in decoder.state_dict().items():
# 	print(i, k.size(), torch.numel(k))
# 	cnt += torch.numel(k)
# print(cnt)

# ll.mean().backward()
# print(decoder.Wk1.weight.grad)
# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634