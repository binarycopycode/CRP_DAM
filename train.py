import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from datetime import datetime

from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data, Generator,data_from_txt
from config import Config, load_pkl, train_parser

def train(cfg, log_path = None):
    #将会让程序在开始torch.save(model.state_dict(), '%s%s_epoch%s.pt' % (cfg.weight_dir, cfg.task, epoch))时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # 进而实现网络的加速
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    log_path = '%s%s_%s.txt' % (cfg.log_dir, cfg.task, cfg.dump_date)  # cfg.log_dir = ./Csv/
    start_t=time()
    # open w就是覆盖，a就是在后面加 append
    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%y%m%d_%H_%M'))
    with open(log_path, 'a') as f:
        f.write('\n start training \n')

    with open(log_path, 'a') as f:
        f.write(''.join('%s: %s\n' % item for item in vars(cfg).items()))
    #t1=time()
    #dataset = Generator(device, cfg.batch*cfg.batch_steps,
    #                    cfg.n_containers, cfg.max_stacks, cfg.max_tiers)

    #t2=time()
    #print('generate data: %dmin%dsec'%((t2-t1)//60,(t2-t1)%60)) 生成一次数据1分46s

    #print('233')

    model = AttentionModel(device=device,embed_dim=cfg.embed_dim,n_encode_layers= cfg.n_encode_layers,n_heads= cfg.n_heads,tanh_clipping= cfg.tanh_clipping
                           ,n_containers=cfg.n_containers,max_stacks=cfg.max_stacks,max_tiers=cfg.max_tiers)
    model.train()
    model=model.to(device)

    baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples,
                               cfg.embed_dim,cfg.n_containers, cfg.max_stacks,cfg.max_tiers , cfg.warmup_beta, cfg.wp_epochs, device,log_path)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    #bs batch steps number of samples = batch * batch_steps
    def rein_loss(model, inputs, bs, t, device):
        # ~ inputs = list(map(lambda x: x.to(device), inputs))

        # decode_type是贪心找最大概率还是随机采样
        # L(batch) 就是返回的cost ll就是采样得到的路径的概率
        L, ll = model(inputs, decode_type='sampling')
        #b = bs[t] if bs is not None else baseline.eval(inputs, L)
        b=torch.FloatTensor([L.mean()]).to(device)
        return ((L - b) * ll).mean(), L.mean()

    tt1 = time()


    t1=time()
    for epoch in range(cfg.epochs):

        ave_loss, ave_L = 0., 0.

        datat1=time()
        dataset=Generator(device,cfg.batch*cfg.batch_steps,
                          cfg.n_containers,cfg.max_stacks,cfg.max_tiers)
        datat2=time()
        print('data_gen: %dmin%dsec' % ((datat2 - datat1) // 60, (datat2 - datat1) % 60))

        bs=baseline.eval_all(dataset)
        bs = bs.view(-1, cfg.batch) if bs is not None else None  # bs: (cfg.batch_steps, cfg.batch) or None

        model.train()
        dataloader=DataLoader(dataset, batch_size = cfg.batch, shuffle = True)
        for t, inputs in enumerate(dataloader):
            loss,L_mean=rein_loss(model,inputs,bs,t,device)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            # print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
            # https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            ave_loss += loss.item()
            ave_L += L_mean.item()

            if t % (cfg.batch_verbose) == 0:
                t2 = time()
                # //60是对小数取整
                print('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec' % (
                    epoch, t, ave_loss / (t + 1), ave_L / (t + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                # 如果要把日志文件保存下来
                if cfg.islogger:
                    with open(log_path, 'a') as f:
                        f.write('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec \n' % (
                    epoch, t, ave_loss / (t + 1), ave_L / (t + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                t1 = time()

        #print('after this epoch grad: ', model.Decoder.Wk1.weight.grad[0][0])

        # 看是不是要更新baseline
        #这里为了让baseline不变化给model加上eval
        model.eval()
        baseline.epoch_callback(model, epoch)
        #torch.save(model.state_dict(), '%s%s_epoch%s.pt' % (cfg.weight_dir, cfg.task, epoch))

        if epoch==cfg.epochs-1:
            #data = data_from_txt("data/test.txt")
            #data=data.to(device)
            #baseline.model.eval()
            torch.save(baseline.model.state_dict(), '%s%s_epoch%s_2.pt' % (cfg.weight_dir, cfg.task, epoch))
            #torch.save(baseline.model.Decoder.Encoder.state_dict(),'%s%s_encoder_epoch%s.pt' % (cfg.weight_dir, cfg.task, epoch))
            #with torch.no_grad():
                #cost=baseline.rollout(model=baseline.model,dataset=data,batch=40)
            #print('test baseline model')
            #print('test.txt:mean',cost.mean())

    tt2 = time()
    print('all time, %dmin%dsec' % (
        (tt2 - tt1) // 60, (tt2 - tt1) % 60))

if __name__ == '__main__':
	cfg = load_pkl(train_parser().path)

	train(cfg)
    #nohup python -u train.py -p Pkl/CRP_9_3_5_train.pkl >>./Csv/nohup.txt 2>&1 &
