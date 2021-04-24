import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
import numpy as np


class position_embedding(nn.Module):

    def __init__(self, input_length, num_of_vertices, embedding_size, temporal=True, spatial=True, config=None):
        super(position_embedding, self).__init__()
        self.input_length = input_length
        self.num_of_vertices = num_of_vertices
        self.embedding_size = embedding_size
        self.temporal = temporal
        self.spatial = spatial
        self.temporal_emb = torch.zeros((1, input_length, 1, embedding_size))
        if config["gpu"]:
            self.temporal_emb = self.temporal_emb.cuda()
        xavier_uniform(self.temporal_emb)
        self.spatial_emb = torch.zeros((1, 1, num_of_vertices, embedding_size))
        if config["gpu"]:
            self.temporal_emb = self.spatial_emb.cuda()
        xavier_uniform(self.temporal_emb)

    def forward(self, data):
        if self.temporal:
            data += self.temporal_emb
        if self.spatial:
            data += self.spatial_emb
        return data


class gcn_opration(nn.Module):

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

        # TODO 这里和 Ndarray 的乘法一样？
        data = torch.matmul(adj, data)

        if self.activation == "GLU":
            data = self.layer(data)
            lhs, rhs = data.split(self.num_of_features, -1)
            data = lhs * torch.sigmoid(rhs)

        elif self.activation == "relu":
            data = torch.relu(self.layer(data))

        return data


class Stsgcm(nn.Module):

    def __init__(self, filters, num_of_features, num_of_vertices, activation):
        super().__init__()
        self.filters = filters
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.layers = nn.ModuleList()
        for i in range(len(filters)):
            self.layers.append(gcn_opration(filters[i], num_of_features, num_of_vertices, activation))
            num_of_features = filters[i]

    def forward(self, data, adj):
        need_concat = []
        for i in range(len(self.layers)):
            data = self.layers[i](data, adj)
            # TODO 改成这里转一下？
            need_concat.append(torch.transpose(data, 1, 0))

        need_concat = [
            torch.unsqueeze(
                i[self.num_of_vertices:2 * self.num_of_vertices, :, :],
                dim=0
            ) for i in need_concat
        ]

        return torch.max(torch.cat(need_concat, dim=0), dim=0)[0]


class stsgcl(nn.Module):

    def __init__(self, T, num_of_vertices, num_of_features, filters, module_type, activation, temporal_emb=True,
                 spatial_emb=True, config=None):

        super().__init__()
        assert module_type in {'sharing', 'individual'}
        self.T = T
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.module_type = module_type
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        if module_type == 'individual':
            self.layer = Sthgcn_layer_individual(
                T, num_of_vertices, num_of_features, filters,
                activation, temporal_emb, spatial_emb, config
            )
        else:
            self.layer = Sthgcn_layer_sharing(
                T, num_of_vertices, num_of_features, filters,
                activation, temporal_emb, spatial_emb, config
            )

    def forward(self, data, adj):
        return self.layer(data, adj)


class Sthgcn_layer_individual(nn.Module):

    def __init__(self, T, num_of_vertices, num_of_features, filters,
                 activation, temporal_emb=True, spatial_emb=True, config=None):
        super().__init__()
        self.T = T
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = position_embedding(T, num_of_vertices, num_of_features,
                                                     temporal_emb, spatial_emb, config)
        self.gcms = nn.ModuleList()
        for i in range(self.T - 2):
            self.gcms.append(Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                                    activation=self.activation))

    def forward(self, data, adj):
        data = self.position_embedding(data)
        need_concat = []
        for i in range(self.T - 2):
            # shape is (B, 3, N, C)
            t = data[:, i:i + 3, :, :]

            # shape is (B, 3N, C)
            t = torch.reshape(t, (-1, 3 * self.num_of_vertices, self.num_of_features))

            # shape is (3N, B, C)
            # TODO 这里为什么要转一下
            # t = torch.transpose(t, 1, 0)

            # shape is (N, B, C')
            t = self.gcms[i](t, adj)

            # shape is (B, N, C')
            t = torch.transpose(t, 0, 1)

            # shape is (B, 1, N, C')
            need_concat.append(torch.unsqueeze(t, dim=1))

        # shape is (B, T-2, N, C')
        return torch.cat(need_concat, dim=1)


class Sthgcn_layer_sharing(nn.Module):

    def __init__(self, T, num_of_vertices, num_of_features, filters,
                 activation, temporal_emb=True, spatial_emb=True, config=None):
        super().__init__()
        self.T = T
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.position_embedding = position_embedding(T, num_of_vertices, num_of_features,
                                                     temporal_emb, spatial_emb, config)
        self.gcm = Stsgcm(self.filters, self.num_of_features, self.num_of_vertices,
                          activation=self.activation)

    def forward(self, data, adj):
        data = self.position_embedding(data)

        need_concat = []
        for i in range(self.T - 2):
            # shape is (B, 3, N, C)
            t = data[:, i:i + 3, :, :]

            # shape is (B, 3N, C)
            t = torch.reshape(t, (-1, 3 * self.num_of_vertices, self.num_of_features))

            # shape is (3N, B, C)
            t = torch.transpose(t, 0, 1)
            need_concat.append(t)

        # shape is (3N, (T-2)*B, C)
        t = torch.cat(need_concat, dim=1)

        # shape is (N, (T-2)*B, C')
        t = self.gcm(t, adj)

        # shape is (N, T - 2, B, C)
        t = t.reshape((self.num_of_vertices, self.T - 2, -1, self.filters[-1]))

        # shape is (B, T - 2, N, C)
        return torch.transpose(t, 0, 2)


class Output_layer(nn.Module):

    def __init__(self, num_of_vertices, input_length, num_of_features, num_of_filters=128, predict_length=12):
        super().__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length
        # TODO 这里到底是输出 num_of_features 个还是 num_of_filter 个
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


def construct_adj(A, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times of the does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    '''
    N = len(A)
    adj = np.zeros([N * steps] * 2)

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj


class STSGCN(AbstractTrafficStateModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # TODO (Check) config 保存了配置文件信息， feature 保存了数据集特性信息， 需要加载
        self.module_type = config['module_type']
        self.activation = config['act_type']
        self.temporal_emb = config['temporal_emb']
        self.spatial_emb = config['spatial_emb']
        self.use_mask = config['use_mask']
        self.batch_size = config['batch_size']
        self.num_of_vertices = config['num_of_vertices']

        self.input_length = config['points_per_hour']
        self.num_for_predict = config['num_for_predict']

        # TODO （Check）这里有一个 rho 参数、predict_length 参数
        self.predict_length = config['predict_length']
        self.rho = config['rho']

        # TODO（Check） 构建局部临接矩阵
        self.adj = self.data_feature.get("adj_mx")
        self.adj = construct_adj(self.adj, 3)
        self.adj = torch.tensor(self.adj, requires_grad=False, dtype=torch.float32)
        if config["gpu"]:
            self.adj = self.adj.cuda()

        if self.use_mask:
            # TODO（Check） 初始化遮罩矩阵
            self.mask = torch.tensor((self.adj != 0) + 0.0)
            if config["gpu"]:
                self.mask.cuda()

        # TODO (Check) 从 config 取得 filters 建立 滤波层
        self.filter_list = config["filters"]

        # TODO (Check) First_layer_embedding
        self.num_of_features = config['num_of_features']
        first_layer_embedding_size = config["first_layer_embedding_size"]
        if first_layer_embedding_size:
            self.first_layer_embedding = nn.Linear(self.num_of_features, first_layer_embedding_size)
            self.num_of_features = first_layer_embedding_size
        else:
            self.first_layer_embedding = None

        # TODO (Check) 时空同步层建立
        self.stsgcl_layers = nn.ModuleList()
        for idx, filters in enumerate(self.filter_list):
            self.stsgcl_layers.append(stsgcl(self.input_length, self.num_of_vertices,
                                             self.num_of_features, filters, self.module_type,
                                             activation=self.activation,
                                             temporal_emb=self.temporal_emb,
                                             spatial_emb=self.spatial_emb, config=config))
            self.input_length -= 2
            self.num_of_features = filters[-1]

        # TODO (Check) 这里是不是改成 ModuleList
        self.outputs = nn.ModuleList()
        for i in range(self.predict_length):
            self.outputs.append(Output_layer(self.num_of_vertices, self.input_length, self.num_of_features,
                                             num_of_filters=4, predict_length=1))

        # TODO (Check) 定义 Loss 函数
        self.loss = nn.SmoothL1Loss(beta=self.rho)

    def forward(self, batch):

        data = batch['X']

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
        y_predicted = self.predict(batch)
        return self.loss(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)
