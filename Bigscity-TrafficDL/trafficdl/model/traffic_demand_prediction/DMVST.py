from logging import getLogger
import torch
import torch.nn as nn
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel

batch_size = 64
mean_label = 0.0
label_max = 0
label_min = 0
seq_len = 8
hidden_dim = 512
threshold = 10.0
eps = 1e-5
loss_lambda = 10.0
feature_len = 0
local_image_size = 9
cnn_hidden_dim_first = 32
fc_oup_dim = 64
lstm_oup_dim = 512
len_valid_id = 0
gama = 10
width = 12
length = 16
padding_size = local_image_size // 2
number = 0


class SpatialViewConv(nn.Module):
    def __init__(self, inp_channel, oup_channel, kernel_size, stride=1, padding=1):
        super(SpatialViewConv, self).__init__()
        self.inp_channel = inp_channel
        self.oup_channel = oup_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=inp_channel, out_channels=oup_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.batch = nn.BatchNorm2d(oup_channel)
        self.relu = nn.ReLU()

    def forward(self, inp):
        return self.relu(self.batch(self.conv(inp)))


class TemporalView(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(fc_oup_dim, lstm_oup_dim)
        self.fc = nn.Linear(in_features=lstm_oup_dim,
                            out_features=2)

    def forward(self, inp):
        lstm_res, (h, c) = self.lstm(inp)
        return self.fc(h[0])


class DMVST(AbstractTrafficStateModel):
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

        # 对input进行padding来是SpatialView的Local CNN
        self.padding = nn.ZeroPad2d((padding_size, padding_size, padding_size, padding_size))

        # 三层Local CNN
        self.local_conv1 = SpatialViewConv(inp_channel=2, oup_channel=cnn_hidden_dim_first, kernel_size=3, stride=1,
                                           padding=1)
        self.local_conv2 = SpatialViewConv(inp_channel=cnn_hidden_dim_first, oup_channel=cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)
        self.local_conv3 = SpatialViewConv(inp_channel=cnn_hidden_dim_first, oup_channel=cnn_hidden_dim_first,
                                           kernel_size=3, stride=1, padding=1)

        # 全连接降维
        self.fc1 = nn.Linear(in_features=cnn_hidden_dim_first * local_image_size * local_image_size,
                             out_features=fc_oup_dim)

        # TemporalView
        self.temporalLayers = [[TemporalView() for i in range(width)] for j in range(length)]

    def spatial_forward(self, grid_batch):
        # input 9 * 9
        x1 = self.local_conv1(grid_batch)
        x2 = self.local_conv2(x1)
        x3 = self.local_conv3(x2)
        x4 = self.fc1(torch.flatten(x3, start_dim=1))
        return x4

    def forward(self, batch):
        """
        调用模型计算这个batch输入对应的输出，nn.Module必须实现的接口
        :param batch: 输入数据，类字典，可以按字典的方法取数据
        :return:
        """
        # 记录batch number
        global number

        # input转换为卷积运算的格式 (sample, leq, w, h, channel) -> (sample * leq, channel, w, h)
        x = batch['X'].transpose(2, 4)
        shape0 = x.shape[0]
        shape2 = x.shape[2]
        shape3 = x.shape[3]
        shape4 = x.shape[4]
        x = x.reshape((shape0 * seq_len, shape2, shape3, shape4))

        # 对输入进行0填充
        x_padding = self.padding(x)

        # 构造输出
        oup = torch.zeros(batch['y'].shape)

        # 对每个grid进行预测
        for i in range(padding_size, width - padding_size):
            for j in range(padding_size, length - padding_size):
                spatial_res = self.spatial_forward(
                    x_padding[:, :, i - padding_size:i + padding_size + 1, j - padding_size: j + padding_size + 1])
                seq_res = spatial_res.reshape((seq_len, spatial_res.shape[0] // seq_len, spatial_res.shape[1]))
                temporal_res = self.temporalLayers[i - padding_size][j - padding_size](seq_res)
                oup[:, :, i, j, :] = temporal_res.reshape(shape0, 1, 2)
        number += 1
        self._logger.warning("batch-{:d}".format(number))
        # self._scaler.inverse_transform
        # 1.取数据，假设字典中有4类数据，X,y,X_ext,y_ext
        # 当然一般只需要取输入数据，例如X,X_ext，因为这个函数是用来计算输出的
        # 模型输入的数据的特征维度应该等于self.feature_dim
        # x = batch['X']  # shape = (batch_size, input_length, ..., feature_dim)
        # 例如: y = batch['y'] / X_ext = batch['X_ext'] / y_ext = batch['y_ext']]
        # 2.根据输入数据计算模型的输出结果
        # 模型输出的结果的特征维度应该等于self.output_dim
        # 模型输出的结果的其他维度应该跟batch['y']一致，只有特征维度可能不同（因为batch['y']可能包含一些外部特征）
        # 如果模型的单步预测，batch['y']是多步的数据，则时间维度也有可能不同
        # 例如: outputs = self.model(x)
        # 3.返回输出结果
        # 例如: return outputs
        return oup

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
        res = loss.masked_mape_torch(y_predicted, y_true)
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
