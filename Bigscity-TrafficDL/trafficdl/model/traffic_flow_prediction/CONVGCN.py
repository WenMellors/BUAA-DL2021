#18373391 王雨轩 18373304 杨祎然
from logging import getLogger
import torch
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torch.nn.functional as F
import numpy as np
import csv
import math
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # FloatTensor建立tensor
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # 初始化权重
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = []
        for i in range(input.shape[0]):
            temp = torch.mm(input[i],self.weight)
            temp = temp.detach().numpy()
            support.append(temp)
        support = torch.Tensor(support)
        adjT = torch.Tensor(adj)
        #output = torch.spmm(adjT, support)
        output = []
        for i in range(support.shape[0]):
            temp = torch.spmm(adjT, support[i])
            temp = temp.detach().numpy()
            output.append(temp)
        output = torch.Tensor(output)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class CONVGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # 邻接矩阵
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # 网格个数
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        self.len_row = self.data_feature.get('len_row', 1)  # 网格行数
        self.len_column = self.data_feature.get('len_column', 1)  # 网格列数
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.model=None
        self.node_embeddings = nn.Parameter(torch.randn(16), requires_grad=True)


        self.gc11 = GCN(30, 16, 15)
        self.gc12 = GCN(30, 16, 15)
        self.Conv = nn.Conv3d(in_channels=276, out_channels=276, kernel_size=3, stride=(1, 1, 1), padding=(2,2,2))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.fc = nn.Linear(38640, 276)

    def forward(self, batch):
        adj_new = self.adj_mx
        for i in range(276):
            adj_new[i,i] = 1
        x = batch['X']
        in1 = x[:,:,:,0]
        in2 = x[:,:,:,1]
        out1 = self.gc11(in1, adj_new)
        out1 = torch.reshape(out1, (out1.shape[0],276, 5, 3, 1)).detach().numpy()
        out2 = self.gc12(in2, adj_new)
        out2 = torch.reshape(out2, (out2.shape[0],276, 5, 3, 1)).detach().numpy()
        out = np.concatenate([out1, out2], axis=4)
        out = torch.Tensor(out)
        out = self.relu(self.Conv(out))
        #out = self.pool(out)
        out = out.view(-1, 38640)
        out = self.fc(out)
        return out

    def calculate_loss(self,batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        #y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        #y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)
