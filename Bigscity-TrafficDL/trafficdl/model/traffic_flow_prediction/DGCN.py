from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
from scipy.sparse.linalg import eigs
from trafficdl.model import loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import BatchNorm2d, Conv1d, Conv2d, Parameter, LayerNorm, BatchNorm1d


class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size, device):
        super(TATT, self).__init__()
        self.device = device
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True).to(device)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True).to(device)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True).to(device)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze(1)  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(1)  # b,c,l

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        # normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        return coefs


class SATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size, device):
        super(SATT, self).__init__()
        self.device = device
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True).to(device)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True).to(device)

        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True).to(device)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, seq):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)  # b,n,l

        c2 = seq.permute(0, 3, 1, 2)  # b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        # normalization
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
        return coefs


class cheby_conv_ds(nn.Module):
    def __init__(self, c_in, c_out, K, device):
        super(cheby_conv_ds, self).__init__()
        self.device = device
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj, ds):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L0 = torch.eye(nNode).to(self.device)
        L1 = adj

        L = ds * adj
        Im = ds * torch.eye(nNode).to(self.device)
        Ls.append(Im)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            L3 = ds * L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out

    # ASTGCN_block


class ST_BLOCK_0(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, device):
        super(ST_BLOCK_0, self).__init__()
        self.device = device

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT = TATT(c_in, num_nodes, tem_size, device)
        self.SATT = SATT(c_in, num_nodes, tem_size, device)
        self.dynamic_gcn = cheby_conv_ds(c_in, c_out, K, device)
        self.K = K

        self.time_conv = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        T_coef = self.TATT(x)
        T_coef = T_coef.transpose(-1, -2)
        x_TAt = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        S_coef = self.SATT(x)  # B x N x N

        spatial_gcn = self.dynamic_gcn(x_TAt, supports, S_coef)
        spatial_gcn = torch.relu(spatial_gcn)
        time_conv_output = self.time_conv(spatial_gcn)
        out = self.bn(torch.relu(time_conv_output + x_input))

        return out, S_coef, T_coef


# DGCN_Mask&&DGCN_Res
class T_cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt, device):
        super(T_cheby_conv, self).__init__()
        self.device = device
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).to(self.device)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        Lap = Lap.transpose(-1, -2)
        # print(Lap)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out


class ST_BLOCK_1(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, device):
        super(ST_BLOCK_1, self).__init__()
        self.device = device

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT_1 = TATT_1(c_out, num_nodes, tem_size, device)
        self.dynamic_gcn = T_cheby_conv(c_out, 2 * c_out, K, Kt, device)
        self.K = K
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        x_1 = self.time_conv(x)
        x_1 = F.leaky_relu(x_1)
        x_1 = F.dropout(x_1, 0.5, self.training)
        x_1 = self.dynamic_gcn(x_1, supports)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, supports, T_coef


# 2
# DGCN
class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt, device):
        super(T_cheby_conv_ds, self).__init__()
        self.device = device
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape

        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample, 1, 1).to(self.device)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out


class SATT_3(nn.Module):
    def __init__(self, c_in, num_nodes):
        super(SATT_3, self).__init__()
        self.conv1 = Conv2d(c_in * 12, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(c_in * 12, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.bn = LayerNorm([num_nodes, num_nodes, 4])
        self.c_in = c_in

    def forward(self, seq):
        shape = seq.shape
        seq = seq.permute(0, 1, 3, 2).contiguous().view(shape[0], shape[1] * 12, shape[3] // 12, shape[2])
        seq = seq.permute(0, 1, 3, 2)
        shape = seq.shape
        # b,c*12,n,l//12
        f1 = self.conv1(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 3, 1, 4, 2).contiguous()
        f2 = self.conv2(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        logits = torch.einsum('bnclm,bcqlm->bnqlm', f1, f2)
        # a,_ = torch.max(logits, -1, True)
        # logits = logits - a
        # logits = logits.permute(0,2,1,3).contiguous()
        # logits=self.bn(logits).permute(0,3,2,1).contiguous()
        logits = logits.permute(0, 3, 1, 2, 4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits, -1)
        return logits


class SATT_2(nn.Module):
    def __init__(self, c_in, num_nodes):
        super(SATT_2, self).__init__()
        self.conv1 = Conv2d(c_in, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(c_in, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.bn = LayerNorm([num_nodes, num_nodes, 12])
        self.c_in = c_in

    def forward(self, seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 3, 1, 4, 2).contiguous()
        f2 = self.conv2(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        logits = torch.einsum('bnclm,bcqlm->bnqlm', f1, f2)
        # a,_ = torch.max(logits, -1, True)
        # logits = logits - a
        # logits = logits.permute(0,2,1,3).contiguous()
        # logits=self.bn(logits).permute(0,3,2,1).contiguous()
        logits = logits.permute(0, 3, 1, 2, 4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits, -1)
        return logits


class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size, device):
        super(TATT_1, self).__init__()
        A = np.zeros((60, 60))
        for i in range(12):
            for j in range(12):
                A[i, j] = 1
                A[i + 12, j + 12] = 1
                A[i + 24, j + 24] = 1
        for i in range(24):
            for j in range(24):
                A[i + 36, j + 36] = 1
        self.B = (-1e13) * (1 - A)
        self.B = (torch.tensor(self.B)).type(torch.float32).to(device)
        self.device = device
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True).to(device)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True).to(device)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True).to(device)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        # normalization
        # logits=tf_util.batch_norm_for_conv1d(logits, is_training=training,
        #                                   bn_decay=bn_decay, scope='bn')
        # a,_ = torch.max(logits, 1, True)
        # logits = logits - a

        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits + self.B, -1)
        return coefs


class ST_BLOCK_2(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, device):
        super(ST_BLOCK_2, self).__init__()
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT_1 = TATT_1(c_out, num_nodes, tem_size, device)
        self.SATT_3 = SATT_3(c_out, num_nodes)
        self.SATT_2 = SATT_2(c_out, num_nodes)
        self.dynamic_gcn = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt, device)
        self.LSTM = nn.LSTM(num_nodes, num_nodes, batch_first=True)  # b*n,l,c
        self.K = K
        self.tem_size = tem_size
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])
        self.device = device

    def forward(self, x, supports):
        x_input = self.conv1(x)
        x_1 = self.time_conv(x)
        x_1 = F.leaky_relu(x_1)
        x_tem1 = x_1[:, :, :, 0:48]
        x_tem2 = x_1[:, :, :, 48:60]
        S_coef1 = self.SATT_3(x_tem1)
        # print(S_coef1.shape)
        S_coef2 = self.SATT_2(x_tem2)
        # print(S_coef2.shape)
        S_coef = torch.cat((S_coef1, S_coef2), 1)  # b,l,n,c
        shape = S_coef.shape
        # print(S_coef.shape)
        h = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).to(self.device)
        c = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).to(self.device)
        hidden = (h, c)
        S_coef = S_coef.permute(0, 2, 1, 3).contiguous().view(shape[0] * shape[2], shape[1], shape[3])
        S_coef = F.dropout(S_coef, 0.5, self.training)  # 2020/3/28/22:17,试验下效果
        _, hidden = self.LSTM(S_coef, hidden)
        adj_out = hidden[0].squeeze().view(shape[0], shape[2], shape[3]).contiguous()
        adj_out1 = (adj_out) * supports
        x_1 = F.dropout(x_1, 0.5, self.training)
        x_1 = self.dynamic_gcn(x_1, adj_out1)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, adj_out, T_coef


# DGCN_R
class TATT_1_r(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size, device):
        super(TATT_1_r, self).__init__()
        self.device = device
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True).to(self.device)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True).to(self.device)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True).to(self.device)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        # normalization
        # logits=tf_util.batch_norm_for_conv1d(logits, is_training=training,
        #                                   bn_decay=bn_decay, scope='bn')
        # a,_ = torch.max(logits, 1, True)
        # logits = logits - a

        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs


class ST_BLOCK_2_r(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, device):
        super(ST_BLOCK_2_r, self).__init__()
        self.device = device
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT_1 = TATT_1_r(c_out, num_nodes, tem_size, device)
        self.SATT_3 = SATT_3(c_out, num_nodes)
        self.SATT_2 = SATT_2(c_out, num_nodes)
        self.dynamic_gcn = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt, device)
        self.LSTM = nn.LSTM(num_nodes, num_nodes, batch_first=True)  # b*n,l,c
        self.K = K
        self.tem_size = tem_size
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        x_1 = self.time_conv(x)
        x_1 = F.leaky_relu(x_1)
        x_tem1 = x_1[:, :, :, 0:12]
        x_tem2 = x_1[:, :, :, 12:24]
        S_coef1 = self.SATT_3(x_tem1)
        # print(S_coef1.shape)
        S_coef2 = self.SATT_2(x_tem2)
        # print(S_coef2.shape)
        S_coef = torch.cat((S_coef1, S_coef2), 1)  # b,l,n,c
        shape = S_coef.shape
        # print(S_coef.shape)
        h = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).to(self.device)
        c = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).to(self.device)
        hidden = (h, c)
        S_coef = S_coef.permute(0, 2, 1, 3).contiguous().view(shape[0] * shape[2], shape[1], shape[3])
        S_coef = F.dropout(S_coef, 0.5, self.training)  # 2020/3/28/22:17
        _, hidden = self.LSTM(S_coef, hidden)
        adj_out = hidden[0].squeeze().view(shape[0], shape[2], shape[3]).contiguous()
        adj_out1 = (adj_out) * supports
        x_1 = F.dropout(x_1, 0.5, self.training)
        x_1 = self.dynamic_gcn(x_1, adj_out1)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, adj_out, T_coef


# DGCN_GAT
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, length, Kt, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.length = length
        self.alpha = alpha
        self.concat = concat

        self.conv0 = Conv2d(self.in_features, self.out_features, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)

        self.conv1 = Conv1d(self.out_features * self.length, 1, kernel_size=1,
                            stride=1, bias=False)
        self.conv2 = Conv1d(self.out_features * self.length, 1, kernel_size=1,
                            stride=1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        '''
        :param input: 输入特征 (batch,in_features,nodes,length)->(batch,in_features*length,nodes)
        :param adj:  邻接矩阵 (batch,batch)
        :return: 输出特征 (batch,out_features)
        '''
        input = self.conv0(input)
        shape = input.shape
        input1 = input.permute(0, 1, 3, 2).contiguous().view(shape[0], -1, shape[2]).contiguous()

        f_1 = self.conv1(input1)
        f_2 = self.conv1(input1)

        logits = f_1 + f_2.permute(0, 2, 1).contiguous()
        attention = F.softmax(self.leakyrelu(logits) + adj, dim=-1)  # (batch,nodes,nodes)
        # attention1 = F.dropout(attention, self.dropout, training=self.training) # (batch,nodes,nodes)
        attention = attention.transpose(-1, -2)
        h_prime = torch.einsum('bcnl,bnq->bcql', input, attention)  # (batch,out_features)
        return h_prime, attention


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, length, Kt):
        """
        Dense version of GAT.
        :param nfeat: 输入特征的维度
        :param nhid:  输出特征的维度
        :param nclass: 分类个数
        :param dropout: dropout
        :param alpha: LeakyRelu中的参数
        :param nheads: 多头注意力机制的个数
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, length=length, Kt=Kt, dropout=dropout, alpha=alpha, concat=True) for _ in
            range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        fea = []
        for att in self.attentions:
            f, S_coef = att(x, adj)
            fea.append(f)
        x = torch.cat(fea, dim=1)
        # x = torch.mean(x,-1)
        return x, S_coef


class ST_BLOCK_3(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, device):
        super(ST_BLOCK_3, self).__init__()
        self.device = device
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT_1 = TATT_1(c_out, num_nodes, tem_size, device)
        self.dynamic_gcn = GAT(c_out, c_out // 1, 0.5, 0.3, 1, tem_size, Kt)
        self.dynamic_gcn1 = GAT(c_out, c_out // 1, 0.5, 0.3, 1, tem_size, Kt)
        self.K = K

        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        x_1 = self.time_conv(x)
        x_1 = F.leaky_relu(x_1)
        x_1 = F.dropout(x_1, 0.5, self.training)
        filter, S_coef = self.dynamic_gcn(x_1, supports)
        gate, _ = self.dynamic_gcn1(x_1, supports)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, S_coef, T_coef


# Gated-STGCN(IJCAI)
class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt, device):
        super(cheby_conv, self).__init__()
        self.device = device
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).to(self.device)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out


class ST_BLOCK_4(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, device):
        super(ST_BLOCK_4, self).__init__()
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.gcn = cheby_conv(c_out // 2, c_out, K, 1, device)
        self.conv2 = Conv2d(c_out, c_out * 2, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.c_out = c_out
        self.conv_1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                             stride=(1, 1), bias=True)
        # self.conv_2=Conv2d(c_out//2, c_out, kernel_size=(1, 1),
        #                stride=(1,1), bias=True)

    def forward(self, x, supports):
        x_input1 = self.conv_1(x)
        x1 = self.conv1(x)
        filter1, gate1 = torch.split(x1, [self.c_out // 2, self.c_out // 2], 1)
        x1 = (filter1) * torch.sigmoid(gate1)
        x2 = self.gcn(x1, supports)
        x2 = torch.relu(x2)
        # x_input2=self.conv_2(x2)
        x3 = self.conv2(x2)
        filter2, gate2 = torch.split(x3, [self.c_out, self.c_out], 1)
        x = (filter2 + x_input1) * torch.sigmoid(gate2)
        return x


# GRCN(ICLR)
class gcn_conv_hop(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ] - input of one single time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : gcn_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt, device):
        super(gcn_conv_hop, self).__init__()
        self.device = device
        c_in_new = (K) * c_in
        self.conv1 = Conv1d(c_in_new, c_out, kernel_size=1,
                            stride=1, bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode = x.shape

        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).to(self.device)
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcn,knq->bckq', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode)
        out = self.conv1(x)
        return out


class ST_BLOCK_5(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt, device):
        super(ST_BLOCK_5, self).__init__()
        self.device = device
        self.gcn_conv = gcn_conv_hop(c_out + c_in, c_out * 4, K, 1)
        self.c_out = c_out
        self.tem_size = tem_size

    def forward(self, x, supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).to(self.device)
        c = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).to(self.device)
        out = []

        for k in range(self.tem_size):
            input1 = x[:, :, :, k]
            tem1 = torch.cat((input1, h), 1)
            fea1 = self.gcn_conv(tem1, supports)
            i, j, f, o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c = c * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(j)
            new_h = torch.tanh(new_c) * (torch.sigmoid(o))
            c = new_c
            h = new_h
            out.append(new_h)
        x = torch.stack(out, -1)
        return x


def scaled_laplacian(weight):
    """
    compute \tilde{L} (scaled laplacian matrix)

    Args:
        weight(np.ndarray): shape is (N, N), N is the num of vertices

    Returns:
        np.ndarray: shape (N, N)
    """
    assert weight.shape[0] == weight.shape[1]
    diag = np.diag(np.sum(weight, axis=1))
    lap = diag - weight
    lambda_max = eigs(lap, k=1, which='LR')[0].real
    return (2 * lap) / lambda_max - np.identity(weight.shape[0])


def cheb_polynomial(l_tilde, k):
    """
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Args:
        l_tilde(np.ndarray): scaled Laplacian, shape (N, N)
        k(int): the maximum order of chebyshev polynomials

    Returns:
        list(np.ndarray): cheb_polynomials, length: K, from T_0 to T_{K-1}
    """
    num = l_tilde.shape[0]
    cheb_polynomials = [np.identity(num), l_tilde.copy()]
    for i in range(2, k):
        cheb_polynomials.append(2 * l_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials


class DGCN(AbstractTrafficStateModel):
    # def __init__(self,c_in,c_out,num_nodes,week,day,recent,K,Kt):
    def __init__(self, config, data_feature):
        super(DGCN, self).__init__(config, data_feature)
        self.data_feature = data_feature
        c_in = config['c_in']
        c_out = config['c_out']
        week = config['week']
        day = config['day']
        recent = config['recent']
        K = config['K']
        Kt = config['Kt']
        self.device = config.get('device', torch.device('cpu'))

        num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)

        tem_size = week + day + recent
        self.block1 = ST_BLOCK_2(c_in, c_out, num_nodes, tem_size, K, Kt, self.device)
        self.block2 = ST_BLOCK_2(c_out, c_out, num_nodes, tem_size, K, Kt, self.device)
        self.bn = BatchNorm2d(c_in, affine=False)

        self.conv1 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv2 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv3 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.conv4 = Conv2d(c_out, 1, kernel_size=(1, 2), padding=(0, 0),
                            stride=(1, 2), bias=True)

        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True).to(self.device)
        nn.init.uniform_(self.h, a=0, b=0.0001)

        adj_mx = self.data_feature.get('adj_mx')

        self.supports = torch.tensor(scaled_laplacian(adj_mx)).to(self.device)

    def forward(self, x_w, x_d, x_r):
        x_w = self.bn(x_w)
        x_d = self.bn(x_d)
        x_r = self.bn(x_r)
        x = torch.cat((x_w, x_d, x_r), -1)

        A = self.h + self.supports
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        A1 = F.dropout(A, 0.5, self.training)

        x, _, _ = self.block1(x.float(), A1.float())
        x, d_adj, t_adj = self.block2(x.float(), A1.float())

        x1 = x[:, :, :, 0:12]
        x2 = x[:, :, :, 12:24]
        x3 = x[:, :, :, 24:36]
        x4 = x[:, :, :, 36:60]

        x1 = self.conv1(x1).squeeze()
        x2 = self.conv2(x2).squeeze()
        x3 = self.conv3(x3).squeeze()
        x4 = self.conv4(x4).squeeze()  # b,n,l
        x = x1 + x2 + x3 + x4
        return x, d_adj, A

    def predict(self, batch):
        """
        Args:
            batch (Batch): a batch of input

        Returns:
            torch.tensor: predict result of this batch
        """
        x = batch['X']
        # (B, Tw+Td+Th, N_nodes, F_in)
        # 时间维度(第1维)上的顺序是CPT，即
        # [0, len_closeness) -- input1
        # [len_closeness, len_closeness+len_period) -- input2
        # [len_closeness+len_period, len_closeness+len_period+len_trend) -- input3
        x_w = None
        x_d = None
        x_r = None
        x = x.permute(0, 3, 2, 1)
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            x_r = x[:, :, :, begin_index:end_index]
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            x_d = x[:, :, :, begin_index:end_index]
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            x_w = x[:, :, :, begin_index:end_index]
        return self.forward(x_w, x_d, x_r)[0].permute(0, 2, 1).unsqueeze(-1)

    def calculate_loss(self, batch):
        y_true = batch['y'][..., :1]
        y_predicted = self.predict(batch)
        # print(y_true.shape)
        # print(y_predicted.shape)
        # y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        # print(y_true.shape)
        # y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # print(y_predicted.shape)
        return loss.masked_mse_torch(y_predicted, y_true)
