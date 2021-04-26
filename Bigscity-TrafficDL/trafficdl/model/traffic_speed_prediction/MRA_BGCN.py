from logging import getLogger
import torch
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
import torch.nn as nn
import time
from trafficdl.model.traffic_speed_prediction.MRABGCN import MRABGCN, MRA_GCN
from trafficdl.model.traffic_speed_prediction.dealadj import*
import numpy as np

class Encoder_GRU(nn.Module):

    def __init__(self, dim_in_enc, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device, in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        super(Encoder_GRU, self).__init__()
        self.DEVICE = device
        self.dim_in_enc = dim_in_enc
        self.gate = MRABGCN(adj_node, adj_edge, self.dim_in_enc * 2, dim_out_node, dim_out_edge, M, range_K,
                             self.dim_in_enc * 2, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.update = MRABGCN(adj_node, adj_edge, self.dim_in_enc * 2, dim_out_node, dim_out_edge, M,
                               range_K,
                               self.dim_in_enc, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_in_enc, self.dim_in_enc))
        self.b = nn.Parameter(torch.FloatTensor(self.dim_in_enc, ))

    def forward(self, inputs=None, hidden_state=None):
        '''
        :param inputs: (P,B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, seq_len, num_vertice, feature = inputs.shape
        output_inner = []
        if hidden_state is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx = hidden_state
        for index in range(seq_len):
            start1 = time.time()
            if inputs is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            else:
                x = inputs[:, index].squeeze(1)

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            start = time.time()
            gates = self.gate(combined)  # gates: B,N, num_features*4
            # print(time.time() - start)
            resetgate, updategate = torch.split(gates, self.dim_in_enc, dim=2)
            resetgate = torch.sigmoid(resetgate)
            updategate = torch.sigmoid(updategate)
            start = time.time()
            cy = torch.tanh(self.update(torch.cat((x, (resetgate * hx)), 2)))
            # print(time.time() - start)
            hy = updategate * hx + (1.0 - updategate) * cy
            hx = hy
            yt = torch.sigmoid(hy.matmul(self.W) + self.b)
            output_inner.append(yt)
            # print(time.time() - start1)
        output_inner = torch.stack(output_inner, dim=0)
        return yt, hy


class Decoder_GRU(nn.Module):
    def __init__(self, seq_target, dim_in_dec, dim_out_dec, adj_node, adj_edge, dim_out_node, dim_out_edge, M,
                 range_K,
                 device,
                 in_drop=0.0, gcn_drop=0.0, residual=False):
        super(Decoder_GRU, self).__init__()
        self.DEVICE = device
        self.seq_target = seq_target
        self.dim_in_dec = dim_in_dec
        self.dim_out_dec = dim_out_dec
        self.gate = MRABGCN(adj_node, adj_edge, self.dim_in_dec * 2, dim_out_node, dim_out_edge, M, range_K,
                             self.dim_in_dec * 2, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.update = MRABGCN(adj_node, adj_edge, self.dim_in_dec * 2, dim_out_node, dim_out_edge, M,
                               range_K, self.dim_in_dec, device, in_drop=in_drop, gcn_drop=gcn_drop,
                               residual=residual)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_in_dec, self.dim_out_dec))
        self.b = nn.Parameter(torch.FloatTensor(self.dim_out_dec, ))

    def forward(self, inputs=None, hidden_state=None):
        '''
        :param inputs: (B,N,F)
        :param hidden_state: ((B,N,F),(B,N,F))
        :return:
        '''

        batch_size, num_vertice, feature = inputs.shape
        output_inner = []
        if hidden_state is None:
            hx = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
        else:
            hx = hidden_state
        for t in range(self.seq_target):
            if inputs is None:
                x = torch.zeros((batch_size, num_vertice, feature)).to(self.DEVICE)
            else:
                x = inputs

            combined = torch.cat((x, hx), 2)  # B,N, num_features*2
            gates = self.gate(combined)  # gates: B,N, num_features*4
            resetgate, updategate = torch.split(gates, self.dim_in_dec, dim=2)
            resetgate = torch.sigmoid(resetgate)
            updategate = torch.sigmoid(updategate)

            cy = torch.tanh(self.update(torch.cat((x, (resetgate * hx)), 2)))
            hy = updategate * hx + (1 - updategate) * cy
            hx = hy
            yt = torch.sigmoid(hy.matmul(self.W) + self.b)
            output_inner.append(yt)
        res = torch.stack(output_inner).permute(1, 0, 2, 3)
        return res

class MRA_BGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """
        构造模型
        :param config: 源于各种配置的配置字典
        :param data_feature: 从数据集Dataset类的`get_data_feature()`接口返回的必要的数据相关的特征
        """
        # 1.初始化父类（必须）
        super().__init__(config, data_feature)
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        dim_in = config['dim_in']
        dim_out_node = config['dim_out_node']
        dim_out_edge = config['dim_out_edge']
        range_K = config['range_K']
        device = self.device
        in_drop = config['in_drop']
        gcn_drop = config['gcn_drop']
        seq_target = config['seq_target']
        dim_out = config['dim_out']
        residual = config['residual']
        adj_node = self.data_feature.get('adj_mx')
        M = get_M(adj_node)
        M = torch.from_numpy(M).type(torch.FloatTensor).to(device)
        adj_edge = get_adj_edge(adj_node)

        self.linear_in = nn.Linear(1, dim_in)
        self.Encoder = Encoder_GRU(dim_in, adj_node, adj_edge, dim_out_node, dim_out_edge, M, range_K, device,
                                   in_drop=in_drop,
                                   gcn_drop=gcn_drop, residual=residual)
        self.Decoder = Decoder_GRU(seq_target, dim_in, dim_out, adj_node, adj_edge, dim_out_node, dim_out_edge, M,
                                   range_K, device, in_drop=in_drop, gcn_drop=gcn_drop, residual=residual)
        self.linear_out = nn.Linear(dim_out, 1)


    def forward(self, batch):
        """
        调用模型计算这个batch输入对应的输出，nn.Module必须实现的接口
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return:
        """
        inputs = torch.Tensor(batch['X'])
        inputs = self.linear_in(inputs)
        output_enc, encoder_hidden_state = self.Encoder(inputs)
        output = self.Decoder(output_enc, encoder_hidden_state)
        output = self.linear_out(output)
        return output


    def calculate_loss(self, batch):
        """
        输入一个batch的数据，返回训练过程这个batch数据的loss，也就是需要定义一个loss函数。
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: training loss (tensor)
        """
        # 1.取出真值 ground_truth
        y_true = batch['y']
        # 2.取出预测值
        y_predicted = self.predict(batch)
        # 3.使用self._scaler将进行了归一化的真值和预测值进行反向归一化（必须）
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # 4.调用loss函数计算真值和预测值的误差
        # trafficdl/model/loss.py中定义了常见的loss函数
        # 如果模型源码用到了其中的loss，则可以直接调用，以MSE为例:
        res = loss.masked_mae_torch(y_predicted, y_true)
        # 如果模型源码所用的loss函数在loss.py中没有，则需要自己实现loss函数
        # ...（自定义loss函数）
        # 5.返回loss的结果
        return res

    def predict(self, batch):
        """
        输入一个batch的数据，返回对应的预测值，一般应该是**多步预测**的结果
        一般会调用上边定义的forward()方法
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return: predict result of this batch (tensor)
        """
        # 如果self.forward()的结果满足要求，可以直接返回
        # 如果不符合要求，例如self.forward()进行了单时间步的预测，但是模型训练时使用的是每个batch的数据进行的多步预测，
        # 则可以参考trafficdl/model/traffic_speed_prediction/STGCN.py中的predict()函数，进行多步预测
        # 多步预测的原则是: 先进行一步预测，用一步预测的结果进行二步预测，**而不是使用一步预测的真值进行二步预测!**
        # 以self.forward()的结果符合要求为例:
        return self.forward(batch)
