from trafficdl.model.abstract_model import AbstractModel
from trafficdl.data.batch import Batch

from torch.nn import LSTM, Linear, Module, Softmax
from torch import split, exp, log, add, sub, mean, sum
from torch import hstack, float32
from math import sqrt


class PreLinear(Module):
    """ PreLinear 输入数据预处理层, 位于 LSTMoy 之前. """

    def __init__(self, in_size, out_size, bias=False):
        """ 初始化 PreLinear 模块对象.

        Args:
            in_size (int): 输入大小
            out_size (int): 输出大小
            bias (bool): 是否增加偏移值
        """

        super(PreLinear, self).__init__()
        self.in_size, self.out_size, self.bias = in_size, out_size, bias
        self.linear_v = Linear(in_features=in_size, out_features=out_size, bias=bias)
        self.linear_l = Linear(in_features=in_size, out_features=out_size, bias=bias)
        self.linear_t = Linear(in_features=in_size, out_features=out_size, bias=bias)

    def forward(self, inp_v, inp_l, inp_t):
        """ 正向传播.

        Args:
            inp_v (torch.tensor.Tensor): shape (batch, seq_len, size)
            inp_l (torch.tensor.Tensor): shape (batch, seq_len, size)
            inp_t (torch.tensor.Tensor): shape (batch, seq_len, size)

        Returns:
            (torch.tensor.Tensor): in shape (batch, seq_len, size)
        """

        return self.linear_v(inp_v) + self.linear_l(inp_l) + self.linear_t(inp_t)


class OutLayer(Module):
    """ OutLayer 输出层, 位于 Attn 层之后. """

    def __init__(self, in_size, out_size, bias=False):
        """ 初始化.

        Args:
            in_size (int): 输入大小
            out_size (int): 输出大小
            bias (bool): 是否增加偏移值
        """

        super(OutLayer, self).__init__()
        self.linear_dyn = Linear(in_features=in_size, out_features=out_size, bias=bias)
        self.linear_sta = Linear(in_features=in_size, out_features=out_size, bias=bias)

    def forward(self, dyn, sta, inp):
        """ 前向传播.

        Args:
            dyn (torch.tensor.Tensor): shape (batch, size) 动态特征描述，来自 Attn 层的输出
            sta (torch.tensor.Tensor): shape (batch, size) 静态特征描述，来自模型输入
            inp (torch.tensor.Tensor): shape (batch, seq_len, size)
                来自 PreLinear 输出的第二部分，用于计算兴趣点可能性预测值

        Returns:
            (torch.tensor.Tensor): shape (batch, seq_len) 兴趣点可能性预测值
        """
        a = (self.linear_dyn(dyn) + self.linear_sta(sta))
        a = a.reshape(a.shape[0], 1, 1, a.shape[1])
        b = inp.reshape(inp.shape[0], inp.shape[1], inp.shape[2], 1)
        c = a.matmul(b)
        return c.reshape(c.shape[0], c.shape[1])


class Attn(Module):
    """ Attention 注意力机制模块, 对 LSTM 中间层输出做加权平均. """

    def __init__(self, size, bias=False):
        """ 初始化.

        Args:
            size (int): 中间层输出向量的大小
            bias: (bool): 是否增加偏移值
        """

        super(Attn, self).__init__()
        self.sqrt_rec_size = 1. / sqrt(size)
        self.linear = Linear(in_features=size, out_features=1, bias=bias)
        self.softmax = Softmax(dim=2)

    def forward(self, inp):
        """ 前向传播.

        Args:
            inp (torch.tensor.Tensor): shape (batch, seq_len, size) 中间层输出序列

        Returns:
            (torch.tensor.Tensor): shape (batch, size)
        """
        w = self.linear(inp).mul(self.sqrt_rec_size)
        w = self.softmax(w.reshape(w.shape[0], 1, w.shape[1]))
        c = w.matmul(inp)
        return c.reshape(c.shape[0], c.shape[2])


class ATSTLSTM(AbstractModel):
    """ ATST_LSTM 轨迹下一跳预测模型. """

    def __init__(self, config, data_feature):
        """ 模型初始化.

        Args:
            config: useless
            data_feature: useless
        """

        super(ATSTLSTM, self).__init__(config, data_feature)
        self.clock = 0
        # 设置输入数据预处理层 PreLinear 参数
        pl_in_size = 1  # 数据处理为1维的
        pl_out_size = pl_in_size
        pl_bias = False
        # 设置 LSTM 参数
        lstm_input_size = pl_out_size  # 必须与 pl_out_size 一致
        lstm_hidden_size = lstm_input_size
        lstm_num_layers = 1
        lstm_bias = True
        lstm_batch_first = True  # 必须为 True, 否则中间运算过程出错
        lstm_dropout = 0
        # 设置 Attn 参数
        at_size = lstm_hidden_size  # 必须与 lstm_hidden_size 一致
        at_bias = False
        # 设置 OutLayer 参数
        ol_in_size = pl_out_size  # 必须与 pl_out_size 一致
        ol_out_size = ol_in_size
        ol_bias = False

        # 构建网络
        self.pre_linear = PreLinear(in_size=pl_in_size, out_size=pl_out_size, bias=pl_bias)
        self.lstm = LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                         num_layers=lstm_num_layers, bias=lstm_bias,
                         batch_first=lstm_batch_first, dropout=lstm_dropout)
        self.attn = Attn(size=at_size, bias=at_bias)
        self.out_layer = OutLayer(in_size=ol_in_size, out_size=ol_out_size, bias=ol_bias)

    def forward(self, inp_v, inp_l, inp_t, sta, nxt):
        """ 前向传播.

        Args:
            inp_v (torch.tensor.Tensor): shape (batch, seq_len_i, size)
            inp_l (torch.tensor.Tensor): shape (batch, seq_len_i, size)
            inp_t (torch.tensor.Tensor): shape (batch, seq_len_i, size)
            sta (torch.tensor.Tensor): shape (batch, size) 静态特征描述
            nxt (int): 下一跳兴趣点位置，相当于以往兴趣点长度
                inp[:, :nxt, :] 输入 LSTM 的以往兴趣点序列
                inp[:, nxt:, :] 输入 OutLayer 的预测兴趣点序列，其中第一个是正确的下一跳兴趣点

        Returns:
            (torch.tensor.Tensor): shape (batch, seq_len) 下一跳兴趣点可能性预测值
        """

        # print(inp_v.shape, sta.shape, nxt)
        inp = self.pre_linear(inp_v, inp_l, inp_t)
        # print(inp.shape)
        inp_i, inp_o = split(inp, [nxt, inp.shape[1] - nxt], dim=1)
        # print(inp_i.shape, inp_o.shape)
        dyn = self.attn(self.lstm(inp_i)[0])
        # print(dyn.shape, sta.shape)
        out = self.out_layer(dyn, sta, inp_o)
        # print('FORWARD:', mean(inp), mean(dyn), mean(out))
        return out

    def _unpack_batch(self, batch):
        """ 拆解 batch.

        Args:
            batch (trafficdl.data.batch): 输入

        Returns:
            inp_v (torch.tensor.Tensor): shape (batch, seq_len_i, size)
            inp_l (torch.tensor.Tensor): shape (batch, seq_len_i, size)
            inp_t (torch.tensor.Tensor): shape (batch, seq_len_i, size)
            sta (torch.tensor.Tensor): shape (batch, size) 静态特征描述
            nxt (int): 下一跳兴趣点位置，相当于以往兴趣点长度
                inp[:, :nxt, :] 输入 LSTM 的以往兴趣点序列
                inp[:, nxt:, :] 输入 OutLayer 的预测兴趣点序列，其中第一个是正确的下一跳兴趣点
        """
        batch: Batch
        nxt = len(batch['loc'][0]) - 1
        inp_v = hstack([batch['loc'], batch['loc_neg']]).type(float32)
        inp_v = inp_v.reshape(inp_v.shape[0], inp_v.shape[1], 1)
        inp_l = hstack([batch['dis'], batch['dis_neg']]).type(float32)
        inp_l = inp_l.reshape(inp_l.shape[0], inp_l.shape[1], 1)
        inp_t = hstack([batch['tim'], batch['tim_neg']]).type(float32)
        inp_t = inp_t.reshape(inp_t.shape[0], inp_t.shape[1], 1)
        sta = batch['uid'].type(float32)
        sta = sta.reshape(sta.shape[0], 1)
        return inp_v, inp_l, inp_t, sta, nxt

    def predict(self, batch):
        """ 预测, 计算下一跳兴趣点可能性预测值.

        Args:
            batch (trafficdl.data.batch): 输入

        Returns:
            (torch.tensor.Tensor): shape (batch, seq_len) 下一跳兴趣点可能性预测值
        """

        inp_v, inp_l, inp_t, sta, nxt = self._unpack_batch(batch)
        # print('INPUT BATCH:', inp_v.shape[0])
        scores = self.forward(inp_v, inp_l, inp_t, sta, nxt)
        return scores

    def calculate_loss(self, batch):
        """ 计算模型损失（不包含正则项）

        Args:
            batch (trafficdl.data.batch): 输入

        Returns:
            (torch.tensor.Tensor): shape () 损失
        """

        score = self.predict(batch)
        score_right, score_others = split(score, [1, score.shape[1] - 1], dim=1)
        loss = sum(log(add(exp(sub(score_others, score_right)), 1)))
        self.clock += 1
        if self.clock % 100 == 0:
            print('ATSTLSTM LOSS:', float(loss))
        return loss
