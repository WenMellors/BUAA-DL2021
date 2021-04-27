from logging import getLogger
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.linalg import eigs
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel

def Scaled_Laplacian(W):
    W = W.astype(float)
    n = np.shape(W)[0]
    d = []
    #simple graph, W_{i,i} = 0
    L = -W
    #get degree matrix d and Laplacian matrix L
    for i in range(n):
        d.append(np.sum(W[i, :]))
        L[i, i] = d[i]
    #symmetric normalized Laplacian L
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])

    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    # lambda_max \approx 2.0
    # we can replace this sentence by setting lambda_max = 2
    return np.matrix(2 * L / lambda_max - np.identity(n))

def Cheb_Poly(L, Ks):
    assert L.shape[0] == L.shape[1]
    n = L.shape[0]
    L0 = np.matrix(np.identity(n))
    L1 = np.matrix(np.copy(L))
    L_list = [np.copy(L0), np.copy(L1)]
    for i in range(1, Ks):
        Ln = np.matrix(2 * L * L1 - L0)
        L_list.append(np.copy(Ln))
        L0 = np.matrix(np.copy(L1))
        L1 = np.matrix(np.copy(Ln))
    # L_lsit (Ks, n*n), Lk (n, Ks*n)
    return np.concatenate(L_list, axis=-1)

class ConvST(nn.Module):
    def __init__(self, supports, k, dim_in, dim_out):
        super(ConvST, self).__init__()
        self.supports = supports
        self.k = k
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x):
        self.B = x.shape[0]
        self.C = x.shape[1]
        self.h = x.shape[2]  # sequence length
        self.N = x.shape[3]  # num of regions
        assert self.C == self.dim_in
        if self.dim_in > self.dim_out:
            w_input = nn.init.xavier_uniform_(torch.empty(self.dim_out, self.dim_in, 1, 1))
            res_input = F.conv2d(x, w_input, strides=1, padding=0)
        elif self.dim_in < self.dim_out:
            res_input = torch.cat((x, torch.zeros(self.B, self.dim_out-self.dim_in, self.h, self.N)), dim=1)
        else:
            res_input = x
        # padding zero
        padding = torch.zeros(self.B, self.dim_in, self.k - 1, self.N)
        # extract spatial-temporal relationships at the same time
        x = torch.cat((x, padding), dim=2)
        # inputs.shape = [B, C, h+k-1, N]
        x = torch.stack([x[:, :, i:i + self.k, :] for i in range(0, self.h)], dim=2)
        # inputs.shape = [B, N, k*dim_in]
        x = torch.reshape(x, (-1, self.N, self.k * self.dim_in))

        conv_out_1 = self.graph_conv(x, self.supports, self.k * self.dim_in, 2 * self.dim_out)
        conv_out_1 = torch.reshape(conv_out_1, [-1, 2*self.dim_out, self.h, self.N])
        conv_out_2 = self.graph_conv(x, self.supports, self.k * self.dim_in, 2 * self.dim_out)
        conv_out_2 = torch.reshape(conv_out_2, [-1, 2*self.dim_out, self.h, self.N])
        out = (conv_out_1[:, 0:self.dim_out, :, :] + res_input) * \
              F.sigmoid(conv_out_2[:, self.dim_out:2*self.dim_out, :, :])
        return out

    def graph_conv(self, inputs, supports, dim_in, dim_out):
        '''
            :param inputs: a tensor of shape [batch, num_nodes, features]
            :param supports: [num_nodes, num_nodes*(order+1)], calculate the chebyshev polynomials in advance to save time
        '''
        dtype = inputs.dtype
        N = inputs.shape[1]
        assert N == supports.shape[0]
        assert dim_in == inputs.shape[2]
        # in fact order is order-1
        order = int(supports.shape[1] / N)
        x_new = torch.reshape(inputs.permute(0, 2, 1), (-1, N))
        x_new = torch.reshape(torch.matmul(x_new, supports), (-1, dim_in, order, N))
        x_new = x_new.permute(0, 3, 1, 2)
        x_new = torch.reshape(x_new, (-1, order * dim_in))
        weights = nn.init.xavier_uniform_(torch.empty((dim_out, order * dim_in), dtype=dtype))
        biases = torch.zeros(dim_out, dtype=dtype)
        outputs = F.linear(x_new, weights, biases)
        outputs = torch.reshape(outputs, [-1, N, dim_out])
        return outputs


class Attention(nn.Module):
    def __init__(self, query, type):
        super(Attention, self).__init__()
        self.query = query  # [B, Et]
        self.type = type

    def forward(self, x):
        # temporal attention: x.shape = [B, d_out, T, N]
        if self.type == 't':
            Et = self.query.shape[1]
            T = x.shape[2]  # Time Sequence
            N = x.shape[3]
            d_out = x.shape[1]

            x_in = torch.reshape(x, (-1, N*d_out, T))
            # x_in.shape = [B, N*d_out, T]
            x = x_in.permute(2, 0, 1)
            # x.shape = [T, B, N*d_out]
            W_3 = nn.init.xavier_uniform_(torch.empty((T, N*d_out, 1), dtype=torch.float32))
            W_4 = nn.init.xavier_uniform_(torch.empty((Et, T), dtype=torch.float32))
            b_1 = torch.zeros(T)
            score = torch.reshape(torch.matmul(x, W_3), (-1, T)) +\
                    torch.matmul(self.query, W_4)+\
                    b_1
            score = F.softmax(F.tanh(score), dim=1)
            # score.shape = [B, T]
            x = torch.matmul(x_in, torch.unsqueeze(score, dim=-1))
            # x.shape = [B, N*d_out, 1]
            x = torch.reshape(x, (-1, d_out, 1, N))
            # x.shape = [B, d_out, 1, N]
            return x
        # channel attention: x.shape = [B, d_out, 1, N]
        elif self.type == 'c':
            Et = self.query.shape[1]
            N = x.shape[3]
            d_out = x.shape[1]

            x_in = torch.reshape(x, (-1, N, d_out))
            # x_in.shape = [B, N, d_out]
            x = x_in.permute(2, 0, 1)
            # x.shape = [d_out, B, N]
            W_5 = nn.init.xavier_uniform_(torch.empty((d_out, N, 1), dtype=torch.float32))
            W_6 = nn.init.xavier_uniform_(torch.empty((Et, d_out), dtype=torch.float32))
            b_2 = torch.zeros(d_out)
            score = torch.reshape(torch.matmul(x, W_5), (-1, d_out)) +\
                    torch.matmul(self.query, W_6) +\
                    b_2
            score = F.softmax(F.tanh(score), dim=1)
            # score.shape = [B, d_out]
            x = torch.matmul(x_in, torch.unsqueeze(score, dim=-1))
            # x.shape = [B, N, 1]
            return x


class TemplateTSP(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        """
        构造模型
        :param config: 源于各种配置的配置字典
        :param data_feature: 从数据集Dataset类的`get_data_feature()`接口返回的必要的数据相关的特征
        """
        # 1.初始化父类（必须）
        super().__init__(config, data_feature)
        # 2.从data_feature获取想要的信息，注意不同模型使用不同的Dataset类，其返回的data_feature内容不同（必须）
        # 以TrafficStateGridDataset为例演示取数据，可以取出如下的数据，用不到的可以不取
        # **这些参数的不能从config中取的**
        self.adj_mx = self.data_feature.get('adj_mx')
        self.supports = torch.tensor(Cheb_Poly(Scaled_Laplacian(self.adj_mx), 2), dtype=torch.float32)
        # 3.初始化log用于必要的输出（必须）
        self._logger = getLogger()
        # 4.初始化device（必须）
        self.device = config.get('device', torch.device('cpu'))
        # 5.初始化输入输出时间步的长度（非必须）
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        # 6.从config中取用到的其他参数，主要是用于构造模型结构的参数（必须）
        # 这些涉及到模型结构的参数应该放在trafficdl/config/model/model_name.json中（必须）
        # 例如: self.blocks = config['blocks']
        self.config = config
        self.C, self.O = config['closeness_sequence_length'], config['nb_flow']
        self.H, self.W, = config['map_height'], config['map_width']
        self._scaler = self.data_feature.get('scaler')
        self.output_dim = self.data_feature.get('output_dim', 1)
        Et = config['et_dim']
        self.horizon = config['horizon']

        self.c_inp = torch.autograd.Variable(torch.ones((128, self.C, self.H, self.W, self.O)), requires_grad=True, name='c_inp')
        self.et_inp = torch.autograd.Variable(torch.ones((128, self.horizon, Et)), requires_grad=True, name='et_inp')
        self.labels = torch.autograd.Variable(torch.ones((128, self.horizon, self.H, self.W, self.O)), requires_grad=True, name='label')
        #self.c_inp = torch.placeholder(torch.float32, [None, C, H, W, O], name='c_inp')
        #self.et_inp = torch.placeholder(torch.float32, (None, horizon, Et), name='et_inp')
        #self.labels = torch.placeholder(torch.float32, shape=[None, horizon, H, W, O], name='label')
        self.labels = torch.reshape(self.labels, (-1, self.horizon, self.H * self.W, self.O))
        # 7.构造深度模型的层次结构（必须）
        # 例如: 使用简单RNN: self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.long_term_layer = nn.Sequential(
            ConvST(self.supports, k=3, dim_in=self.O, dim_out=32),
            nn.BatchNorm2d(32),
            ConvST(self.supports, k=3, dim_in=32, dim_out=32),
            nn.BatchNorm2d(32),
            ConvST(self.supports, k=3, dim_in=32, dim_out=32),
            nn.BatchNorm2d(32),
            ConvST(self.supports, k=3, dim_in=32, dim_out=32),
            nn.BatchNorm2d(32),
            ConvST(self.supports, k=3, dim_in=32, dim_out=32),
            nn.BatchNorm2d(32),
            ConvST(self.supports, k=3, dim_in=32, dim_out=32),
            nn.BatchNorm2d(32),
        )
        self.short_term_gcn = []
        self.attention_t = []
        self.attention_c_1 = []
        self.attention_c_2 = []
        for i in range(self.horizon):
            self.short_term_gcn.append(nn.Sequential(
                ConvST(self.supports, k=3, dim_in=self.O, dim_out=32),
                nn.BatchNorm2d(32),
                ConvST(self.supports, k=3, dim_in=32, dim_out=32),
                nn.BatchNorm2d(32),
                ConvST(self.supports, k=3, dim_in=32, dim_out=32),
                nn.BatchNorm2d(32),
            ))
            self.attention_t.append(
                Attention(self.et_inp[:,i,:], 't')
            )
            self.attention_c_1.append(
                Attention(self.et_inp[:,i,:], 'c')
            )
            self.attention_c_2.append(
                Attention(self.et_inp[:,i,:], 'c')
            )

    def forward(self, x):
        x = torch.reshape(x.data['X'], (-1, self.O, self.C, self.H*self.W))
        l_inputs = self.long_term_layer(x)
        preds = []
        window = 3
        label_padding = x[:, :, -window:, :]
        padded_labels = torch.cat((label_padding, self.labels), dim=2)
        padded_labels = torch.stack([padded_labels[:,:,i:i+window,:] for i in range(0, self.horizon)], dim=2)
        for i in range(0, self.horizon):
            s_inputs = padded_labels[:, :, i, :, :]  # [B, window, N, O]
            s_inputs = self.short_term_gcn[i](s_inputs)
            ls_inputs = torch.cat((s_inputs, l_inputs), dim=1)
            ls_inputs = self.attention_t[i](ls_inputs)
            if self.O == 1:
                pred = self.attention_c_1[i](ls_inputs)
            elif self.O == 2:
                pred = torch.cat((self.attention_c_1[i](ls_inputs),
                                     self.attention_c_2[i](ls_inputs)), dim=-1)
            else:
                print('ERROR WITH NB_FLOW!')
            pred = torch.tile(torch.unsqueeze(pred, -1), (1, 1, 1, 2)).permute(0, 2, 3, 1)
            preds.append(pred)
        
        return torch.stack(preds, dim=1)


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
        print(y_true.shape)
        print(y_predicted.shape)
        res = loss.r2_score_torch(y_predicted, torch.stack(y_true, dim=2))
        # 如果模型源码所用的loss函数在loss.py中没有，则需要自己实现loss函数
        # ...（自定义loss函数）
        # 5.返回loss的结果
        return res

    def predict(self, x):
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
        # x.shape = [B, C, h, N]
        x = torch.ones((128, self.O, self.C, self.H*self.W), dtype=torch.float32)
        l_inputs = self.long_term_layer(x)
        preds = []
        window = 3

        label_padding = x[:, :, -window:, :]
        for i in range(0, self.horizon):
            s_inputs = label_padding
            s_inputs = self.short_term_gcn[i](s_inputs)
            ls_inputs = torch.cat((s_inputs, l_inputs), dim=2)
            ls_inputs = self.attention_t[i](ls_inputs)
            if self.O == 1:
                pred = self.attention_c_1[i](ls_inputs)
            elif self.O == 2:
                pred = torch.cat((self.attention_c_1[i](ls_inputs),
                                  self.attention_c_2[i](ls_inputs)), dim=-1)
            else:
                print('ERROR WITH NB_FLOW!')
            pred = torch.tile(torch.unsqueeze(pred, -1), (1, 1, 1, 2)).permute(0, 2, 3, 1)
            label_padding = torch.cat((label_padding[:, :, 1:, :], pred), dim=2)
            preds.append(pred)

        return torch.stack(preds, dim=1)