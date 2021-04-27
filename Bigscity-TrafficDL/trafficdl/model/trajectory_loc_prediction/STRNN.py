from trafficdl.model.abstract_model import AbstractModel
import torch.nn as nn
import torch
import math
import numpy as np
from torch.autograd import Variable

ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor
up_time = 560632.0  # min
lw_time = 0.
up_dist = 457.335   # km
lw_dist = 0.


class STRNN(AbstractModel):
    """
    请参考开源模型代码，完成本文件的编写。请务必重写 __init__, predict, calculate_loss 三个方法。
    """

    def __init__(self, config, data_feature):
        """
        参数说明：
            config (dict): 配置模块根据模型对应的 config.json 文件与命令行传递的参数
                根据 config 初始化模型参数
            data_feature (dict): 在数据预处理步骤提取到的数据集所属的特征参数，如 loc_size，uid_size 等。
        """

        super(STRNN, self).__init__(config, data_feature)
        self.config = config
        self.hidden_size = 500

        self.weight_ih = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))  # C
        self.weight_th_upper = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))  # T
        self.weight_th_lower = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))  # T
        self.weight_sh_upper = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))  # S
        self.weight_sh_lower = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))  # S

        self.location_weight = nn.Embedding(data_feature['loc_size'], self.hidden_size)
        self.permanet_weight = nn.Embedding(data_feature['uid_size'], self.hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.h0 = Variable(torch.randn(self.hidden_size, 1), requires_grad=False).type(ftype)

        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(self, batch):
        """
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            score (pytorch.tensor): 对应张量 shape 应为 batch_size *
                loc_size。这里返回的是模型对于输入当前轨迹的下一跳位置的预测值。
        """
        user = batch['uid']
        td = batch['current_tim']
        ld = batch['current_dis']
        loc = batch['current_loc']
        td_upper, td_lower, ld_upper, ld_lower, location, hx = self.run(user, td, ld, loc)
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        user_vector = h_tq + torch.t(p_u)
        ret = torch.mm(self.location_weight.weight, user_vector).data.cpu().numpy()
        return ret.T

    def calculate_loss(self, batch):
        """
        参数说明:
            batch (trafficdl.data.batch): 类 dict 文件，其中包含的键值参见任务说明文件。
        返回值:
            loss (pytorch.tensor): 可以调用 pytorch 实现的 loss 函数与 batch['target']
                目标值进行 loss 计算，并将计算结果返回。如模型有自己独特的 loss 计算方式则自行参考实现。
        """
        user = batch['uid']
        td = batch['current_tim']
        ld = batch['current_dis']
        loc = batch['current_loc']

        td_upper, td_lower, ld_upper, ld_lower, location, hx = self.run(user, td, ld, loc)
        h_tq = self.forward(td_upper, td_lower, ld_upper, ld_lower, loc, hx)
        p_u = self.permanet_weight(user)
        q_v = self.location_weight(batch['target'])
        output = torch.mm(q_v, (h_tq + torch.t(p_u)))
        return torch.log(1 + torch.exp(torch.neg(output))).sum()

    def forward(self, td_upper, td_lower, ld_upper, ld_lower, loc, hx):  # 前向传播
        loc_len = min(len(loc), len(td_upper), len(ld_upper))
        Ttd = [((self.weight_th_upper * td_upper[i] + self.weight_th_lower * td_lower[i])
                / (td_upper[i] + td_lower[i])) for i in range(loc_len)]
        Sld = [((self.weight_sh_upper * ld_upper[i] + self.weight_sh_lower * ld_lower[i])
                / (ld_upper[i] + ld_lower[i])) for i in range(loc_len)]

        loc = self.location_weight(loc).view(-1, self.hidden_size, 1)
        loc_vec = torch.sum(torch.cat([torch.mm(Sld[i], torch.mm(Ttd[i], loc[i]))
                                      .view(1, self.hidden_size, 1) for i in range(loc_len)], dim=0), dim=0)
        usr_vec = torch.mm(self.weight_ih, hx)
        hx = loc_vec + usr_vec  # hidden_size x 1
        return self.sigmoid(hx)

    def run(self, user, td, ld, loc):
        seqlen = len(loc)
        # neg_loc = Variable(torch.FloatTensor(1).uniform_(0, len(poi2pos)-1).long()).type(ltype)
        # (neg_lati, neg_longi) = poi2pos.get(neg_loc.data.cpu().numpy()[0])
        rnn_output = self.h0
        for idx in range(seqlen - 1):
            td_upper = Variable(torch.from_numpy(np.asarray((up_time - td[idx]).cpu()))).type(ftype)
            td_lower = Variable(torch.from_numpy(np.asarray((td[idx] - lw_time).cpu()))).type(ftype)
            ld_upper = Variable(torch.from_numpy(np.asarray((up_dist - ld[idx]).cpu()))).type(ftype)
            ld_lower = Variable(torch.from_numpy(np.asarray((ld[idx] - lw_dist).cpu()))).type(ftype)
            location = Variable(torch.from_numpy(np.asarray(loc[idx].cpu()))).type(ltype)
            rnn_output = self.forward(td_upper, td_lower, ld_upper, ld_lower, location,
                                     rnn_output)  # , neg_lati, neg_longi, neg_loc, step)
        td_upper = Variable(torch.from_numpy(np.asarray((up_time - td[-1]).cpu()))).type(ftype)
        td_lower = Variable(torch.from_numpy(np.asarray((td[-1] - lw_time).cpu()))).type(ftype)
        ld_upper = Variable(torch.from_numpy(np.asarray((up_dist - ld[-1]).cpu()))).type(ftype)
        ld_lower = Variable(torch.from_numpy(np.asarray((ld[-1] - lw_dist).cpu()))).type(ftype)
        location = Variable(torch.from_numpy(np.asarray((loc[-1]).cpu()))).type(ltype)

        return td_upper, td_lower, ld_upper, ld_lower, location, rnn_output
