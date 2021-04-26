import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
import numpy as np


# 时空嵌入矩阵，真正的时空特征的嵌入表示
class PositionEmbedding(nn.Module):

    def __init__(self, input_length, num_of_vertices, embedding_size, temporal=True, spatial=True, config=None):
        super(PositionEmbedding, self).__init__()
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.embedding_size = embedding_size
        self.temporal = temporal
        self.spatial = spatial
        self.temporal_emb = torch.zeros((1, input_length, 1, embedding_size))
        # shape is (1, T, 1, C)
        if config["gpu"]:
            self.temporal_emb = self.temporal_emb.cuda()
        xavier_uniform(self.temporal_emb)
        self.spatial_emb = torch.zeros((1, 1, num_of_vertices, embedding_size))
        # shape is (1, 1, N, C)
        if config["gpu"]:
            self.temporal_emb = self.spatial_emb.cuda()
        xavier_uniform(self.temporal_emb)

    def forward(self, data):
        if self.temporal:
            data += self.temporal_emb
        if self.spatial:
            data += self.spatial_emb
        return data


# 图卷积，没啥好说的
class GcnOperation(nn.Module):

    def __init__(self, num_of_filter, num_of_features, num_of_vertices, activation):

        super().__init__()
        assert activation in {'GLU', 'relu'}

        self.num_of_filter = num_of_filter
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        if activation == "GLU":
            self.layer = nn.Linear(num_of_features, 2 * num_of_features)
        elif activation == "relu":
            self.layer = nn.Linear(num_of_features, num_of_features)

    def forward(self, data, adj):

        data = torch.matmul(adj, data)

        if self.activation == "GLU":
            data = self.layer(data)
            lhs, rhs = data.split(self.num_of_features, -1)
            data = lhs * torch.sigmoid(rhs)

        elif self.activation == "relu":
            data = torch.relu(self.layer(data))

        return data


# 同步时空卷积模块，捕获连续 3 个时间片的时空特征
class Stsgcm(nn.Module):

    def __init__(self, filters, num_of_features, num_of_vertices, activation):
        super().__init__()
        self.filters = filters
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            self.layers.append(GcnOperation(filters[i], num_of_features, num_of_vertices, activation))
            num_of_features = filters[i]

    def forward(self, data, adj):

        # 多个卷积层叠加，
        need_concat = []
        for i in range(len(self.layers)):
            data = self.layers[i](data, adj)
            need_concat.append(torch.transpose(data, 1, 0))

        # 且每个卷积层的输出都以类似残差网络的形式输入聚合层
        need_concat = [
            torch.unsqueeze(
                i[self.num_of_vertices:2 * self.num_of_vertices, :, :],
                dim=0
            ) for i in need_concat
        ]

        # 聚合使用最大池化
        return torch.max(torch.cat(need_concat, dim=0), dim=0)[0]


class Stsgcl(nn.Module):

    def __init__(self, t, num_of_vertices, num_of_features, filters, module_type, activation, temporal_emb=True,
                 spatial_emb=True, config=None):

        super().__init__()
        assert module_type in {'sharing', 'individual'}
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.module_type = module_type
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        if module_type == 'individual':
            self.layer = SthgcnLayerIndividual(
                t, num_of_vertices, num_of_features, filters,
                activation, temporal_emb, spatial_emb, config
            )
        else:
            self.layer = SthgcnLayerSharing(
                t, num_of_vertices, num_of_features, filters,
                activation, temporal_emb, spatial_emb, config
            )

    def forward(self, data, adj):
        return self.layer(data, adj)


class SthgcnLayerIndividual(nn.Module):

    def __init__(self, t, num_of_vertices, num_of_features, filters,
                 activation, temporal_emb=True, spatial_emb=True, config=None):
        super().__init__()
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = PositionEmbedding(t, num_of_vertices, num_of_features,
                                                    temporal_emb, spatial_emb, config)

        # 一个 GCM 模块可以捕获连续 3 个时间片的时空特征
        # T 个时间片，每 3 个一捕获，即 T - 2 次，对应 T - 2 个 GCM 模块
        self.gcms = nn.ModuleList()
        for i in range(self.T - 2):
            self.gcms.append(Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                                    activation=self.activation))

    def forward(self, data, adj):
        data = self.position_embedding(data)
        need_concat = []

        for i in range(self.T - 2):
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3, N, C)

            t = torch.reshape(t, (-1, 3 * self.num_of_vertices, self.num_of_features))
            # shape is (B, 3N, C)

            t = self.gcms[i](t, adj)
            # shape is (N, B, C')

            t = torch.transpose(t, 0, 1)
            # shape is (B, N, C')

            need_concat.append(torch.unsqueeze(t, dim=1))
            # shape is (B, 1, N, C')

        # 拼接各个 GCM 模块的输出
        return torch.cat(need_concat, dim=1)
        # shape is (B, T-2, N, C')


# 论文作者消融实验使用，与 Sthgcn_layer_individual 的区别在于，共用一个 stsgcm 模块
class SthgcnLayerSharing(nn.Module):

    def __init__(self, t, num_of_vertices, num_of_features, filters,
                 activation, temporal_emb=True, spatial_emb=True, config=None):
        super().__init__()
        self.T = t
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = PositionEmbedding(t, num_of_vertices, num_of_features,
                                                    temporal_emb, spatial_emb, config)
        self.gcm = Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                          activation=self.activation)

    def forward(self, data, adj):
        data = self.position_embedding(data)

        need_concat = []
        for i in range(self.T - 2):
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3, N, C)

            t = torch.reshape(t, (-1, 3 * self.num_of_vertices, self.num_of_features))
            # shape is (B, 3N, C)

            need_concat.append(t)
            # shape is (B, 3N, C)

        t = torch.cat(need_concat, dim=0)
        # shape is ((T-2)*B, 3N, C)

        t = self.gcm(t, adj)
        # shape is (N, (T-2)*B, C')

        t = t.reshape((self.num_of_vertices, self.T - 2, -1, self.filters[-1]))
        # shape is (N, T - 2, B, C)

        return torch.transpose(t, 0, 2)
        # shape is (B, T - 2, N, C)


class OutputLayer(nn.Module):

    def __init__(self, num_of_vertices, input_length, num_of_features, num_of_filters=128, predict_length=12):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length
        self.hidden_layer = nn.Linear(self.input_length * self.num_of_features, self.num_of_filters)
        self.ouput_layer = nn.Linear(self.num_of_filters, self.predict_length)

    def forward(self, data):
        data = torch.transpose(data, 1, 2)

        # (B, N, T * C)
        data = torch.reshape(
            data, (-1, self.num_of_vertices, self.input_length * self.num_of_features)
        )

        # (B, N, C')
        data = torch.relu(self.hidden_layer(data))

        # (B, N, T')
        data = self.ouput_layer(data)

        # (B, T', N)
        data = torch.transpose(data, 1, 2)

        return data


def construct_adj(a, steps):
    """
    构造局部时空图

    Parameters
    ----------
    a: np.ndarray, shape： (N, N), 原图邻接矩阵

    steps: 时间步长度，原论文是 3 个一合

    Returns
    ----------
    局部时空图矩阵 shape：(N * steps, N * steps)
    """
    n = len(a)
    adj = np.zeros([n * steps] * 2)

    for i in range(steps):
        adj[i * n: (i + 1) * n, i * n: (i + 1) * n] = a

    # 实际就是加了相邻两个时间步节点到自身的边
    for i in range(n):
        for k in range(steps - 1):
            adj[k * n + i, (k + 1) * n + i] = 1
            adj[(k + 1) * n + i, k * n + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


class STSGCN(AbstractTrafficStateModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # 加载配置信息
        self.module_type = config['module_type']
        self.activation = config['act_type']
        self.temporal_emb = config['temporal_emb']
        self.spatial_emb = config['spatial_emb']
        self.use_mask = config['use_mask']
        self.batch_size = config['batch_size']
        self.num_of_vertices = config['num_of_vertices']

        self.input_length = config['points_per_hour']
        self.num_for_predict = config['num_for_predict']

        self.predict_length = config['predict_length']
        self.rho = config['rho']

        # 构造局部时空图
        self.adj = self.data_feature.get("adj_mx")
        self.adj = construct_adj(self.adj, 3)
        self.adj = torch.tensor(self.adj, requires_grad=False, dtype=torch.float32)
        self.num_of_vertices = self.adj.shape[0] // 3
        if config["gpu"]:
            self.adj = self.adj.cuda()

        if self.use_mask:
            # 初始化遮罩矩阵
            self.mask = torch.tensor((self.adj != 0) + 0.0)
            if config["gpu"]:
                self.mask.cuda()

        # 输入嵌入层，增加特征数量，提升表示能力
        self.embedding_dim = config['num_of_features']
        self.num_of_features = config['num_of_features']
        first_layer_embedding_size = config["first_layer_embedding_size"]
        if first_layer_embedding_size:
            self.first_layer_embedding = nn.Linear(self.num_of_features, first_layer_embedding_size)
            self.num_of_features = first_layer_embedding_size
        else:
            self.first_layer_embedding = None

        # 时空同步卷积层组
        self.filter_list = config["filters"]
        self.stsgcl_layers = nn.ModuleList()
        for idx, filters in enumerate(self.filter_list):
            self.stsgcl_layers.append(Stsgcl(self.input_length, self.num_of_vertices,
                                             self.num_of_features, filters, self.module_type,
                                             activation=self.activation,
                                             temporal_emb=self.temporal_emb,
                                             spatial_emb=self.spatial_emb, config=config))
            self.input_length -= 2
            self.num_of_features = filters[-1]

        # 输出层，每个预测时间步一个全连接层

        self.outputs = nn.ModuleList()
        for i in range(self.predict_length):
            self.outputs.append(OutputLayer(self.num_of_vertices, self.input_length, self.num_of_features,
                                            num_of_filters=4, predict_length=1))

        # Huber Loss 损失函数
        self.loss = nn.SmoothL1Loss(beta=self.rho)

    def forward(self, batch):

        data = batch['X']
        # data.shape = (B:batch_size, T:input_length, N:vertical_num, C:feature_num)
        if data.shape[-1] > self.embedding_dim:
            data = data[:, :, :, 0:self.embedding_dim]

        # data.shape = (B, T, N, C:embedding_feature_num)
        if self.first_layer_embedding:
            data = torch.relu(self.first_layer_embedding(data))

        for stsgcl_layer in self.stsgcl_layers:
            data = stsgcl_layer(data, self.mask * self.adj)

        need_concat = []
        for output_layer in self.outputs:
            need_concat.append(output_layer(data))

        data = torch.cat(need_concat, dim=1)

        return data

    def calculate_loss(self, batch):
        y_true = batch['y'][:, :, :, 0]
        y_predicted = self.predict(batch)[:, :, :, 0]
        return self.loss(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch).unsqueeze(-1)
