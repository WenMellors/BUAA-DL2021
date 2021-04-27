from logging import getLogger
import torch
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel


def correntropy(y_true, y_pred):
    alpha = 2
    sigma = 0.3
    lamda = 1 / 0.09
    e = torch.abs(y_true - y_pred)
    return 1 - torch.exp((- lamda) * torch.pow(e, alpha))


def channel_normalization(x):
    max_values, _ = torch.max(torch.abs(x), 2, keepdim=True)
    max_values = max_values + 1e-5
    out = torch.true_divide(x, max_values)
    return out


def wave_net_activation(x):
    tanh_out = torch.tanh(x)
    sigm_out = torch.sigmoid(x)
    return torch.mul(tanh_out, sigm_out)


class residual_block(torch.nn.Module):
    def __init__(self, x, s, i, activation, nb_filters, kernel_size, padding):
        super(residual_block, self).__init__()
        self.modlist = torch.nn.Modulelist([torch.nn.Conv1d(in_channels=21, out_channels=nb_filters,
                                                            kernel_size=kernel_size, dilation=i, padding=padding),
                                            torch.nn.Conv1d(in_channels=21, out_channels=nb_filters, kernel_size=1,
                                                            padding=0)])

    def forward(self, x):
        original_x = x
        x = self.modlist[0](x)
        if activation == 'norm_relu':
            x = torch.nn.ReLU()(x)
            x = channel_normalization(x)
        elif activation == 'wave_net':
            x = wave_net_activation(x)
        x = torch.nn.Dropout(0.2)(x)
        x = self.modlist[1](x)
        res_x = x + original_x
        return res_x, x


class TCN(torch.nn.Module):
    def __init__(self, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=None, activation='norm_relu', padding=0,
                 use_skip_connections=True, dropout_rate=0.2, return_sequences=True):
        super(TCN, self).__init__()
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.activation = activation
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.rb = torch.nn.Modulelist([residual_block(x, s, i, activation, nb_filters, kernel_size, padding)], [
            torch.nn.Conv1d(in_channels=21, out_channels=nb_filters, kernel_size=1, padding=0)])

    def forward(self, inputs):
        if self.dilations is None:
            self.dilations = [1, 2, 4, 8, 16, 32]
        x = inputs
        x = torch.nn.Conv1d(in_channels=21, out_channels=nb_filters, kernel_size=1, padding=0)(x)
        skip_connections = []
        for s in range(self.nb_stacks):
            for i in self.dilations:
                x, skip_out = self.rb[0](x)
                skip_connections.append(skip_out)
        for i in range(len(skip_connections)):
            x = x + skip_connections[i]
        x = torch.nn.ReLU()(x)
        if not self.return_sequences:
            output_slice_index = -1
            x = lambda tt: tt[:, output_slice_index, :](x)
        return x


class TemplateTSP(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        self.adj_mx = self.data_feature.get('adj_mx', 1)  # 邻接矩阵
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # 网格个数
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # 输出维度
        self.len_row = self.data_feature.get('len_row', 1)  # 网格行数
        self.len_column = self.data_feature.get('len_column', 1)  # 网格列数
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
        # ...

        # 7.构造深度模型的层次结构（必须）
        self.modelist = torch.nn.Modulelist(
            [TCN(64, 2, 1, None, 'norm_relu', 0, True, 0.2, True), torch.nn.Linear(64, 276, 30),
             torch.nn.LSTM(10, 20, 2), torch.nn.LSTM(10, 20, 2), torch.nn.Linear(64, 276, 30)])
        # 例如: 使用简单RNN: self.rnn = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, batch):
        """
        调用模型计算这个batch输入对应的输出，nn.Module必须实现的接口
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return:
        """
        # 1.取数据，假设字典中有4类数据，X,y,X_ext,y_ext
        # 当然一般只需要取输入数据，例如X,X_ext，因为这个函数是用来计算输出的
        # 模型输入的数据的特征维度应该等于self.feature_dim
        x = batch['X']  # shape = (batch_size, input_length, ..., feature_dim)
        # 例如: y = batch['y'] / X_ext = batch['X_ext'] / y_ext = batch['y_ext']]
        # 2.根据输入数据计算模型的输出结果
        print(x)
        x = self.modelist[0](x)
        x = self.modelist[1](x)
        x, _ = self.modelist[2](x)
        x, _ = self.modelist[3](x)
        x = torch.nn.Dropout(0.2)(x)
        x = self.modelist[4](x)
        x = torch.sigmoid(x)
        # 模型输出的结果的特征维度应该等于self.output_dim
        # 模型输出的结果的其他维度应该跟batch['y']一致，只有特征维度可能不同（因为batch['y']可能包含一些外部特征）
        # 如果模型的单步预测，batch['y']是多步的数据，则时间维度也有可能不同
        # 例如: outputs = self.model(x)
        # 3.返回输出结果
        return x
        # 例如: return outputs

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        res = torch.mean(correntropy(y_ture, y_predicted))
        return res

    def predict(self, batch):
        return self.forward(batch)
