from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
from logging import getLogger
import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
import numpy as np


def get_spatial_matrix(adj_mx):
    h, w = adj_mx.shape
    inf = float("inf")

    S_near = np.zeros((h, w))
    S_middle = np.zeros((h, w))
    S_distant = np.zeros((h, w))

    i = 0
    for row in adj_mx:
        L_min = np.min(row)
        np.place(row, row == inf, [-1])
        L_max = np.max(row)
        eta = (L_max-L_min)/3
        S_near[i] = np.logical_and(row >= L_min, row < L_min + eta)
        S_middle[i] = np.logical_and(row >= L_min + eta, row < L_min + 2 * eta)
        S_distant[i] = np.logical_and(row >= L_min + 2*eta, row < L_max)
        i = i + 1

    S_near = S_near.astype(np.float32)
    S_middle = S_middle.astype(np.float32)
    S_distant = S_distant.astype(np.float32)
    return tensor(S_near), tensor(S_middle), tensor(S_distant)


class SpatialBlock(nn.Module):
    def __init__(self, n, Smatrix, device):
        super(SpatialBlock, self).__init__()
        self.device = device
        self.S = Smatrix
        self.S = self.S.to(self.device)

        self.linear1 = nn.Linear(n, n)
        self.linear2 = nn.Linear(n, n)

        self.hidden_num = 3

        self.lstm = nn.LSTM(n, n, 1)
        self.lstm2 = nn.LSTM(n, n, 1)

        self.linear3 = nn.Linear(n, n)
        self.linear4 = nn.Linear(n, n)

    def forward(self, x):
        # x:n*c的矩阵
        # (batch,time,node,value)
        batch, time, node, value = x.shape

        # gcn1
        out = self.S.matmul(x.to(self.device))

        out = out.reshape(batch, time, node)
        out = self.linear1(out)
        out = F.relu(out)

        # gcn2
        out = out.reshape(batch, time, node, 1)
        out = self.S.matmul(out.to(self.device))

        out = out.reshape(batch, time, node)
        out = self.linear2(out)
        out = F.relu(out)

        out = out.permute(1, 0, 2)

        # (time,batch,node)
        # LSTM
        out, (a, b) = self.lstm(out)
        out, (a, b) = self.lstm2(out)
        out = out[-1, :, :]

        # Dense
        out = self.linear3(out)
        out = F.relu(out)

        return out


class SpatialComponent(nn.Module):
    def __init__(self, n, adj_mx, len_closeness, device):
        super(SpatialComponent, self).__init__()

        self.device = device

        self.num_nodes = n
        self.len_closeness = len_closeness

        self.near_matrix, self.middle_matrix, self.distant_matrix = get_spatial_matrix(adj_mx)

        self.near_block = SpatialBlock(self.num_nodes, self.near_matrix, self.device)
        self.middle_block = SpatialBlock(self.num_nodes, self.middle_matrix, self.device)
        self.distant_block = SpatialBlock(self.num_nodes, self.distant_matrix, self.device)

        self.linear = nn.Linear(3*n, n)

    def forward(self, x):
        # (batch,time,node,value)
        x = x[:, :self.len_closeness, :, :]

        y_near = self.near_block(x)         # batch*n
        y_middle = self.middle_block(x)     # batch*n
        y_distant = self.distant_block(x)   # batch*n

        out = torch.cat((y_near, y_middle, y_distant), 1)

        out = F.relu(self.linear(out))  # (batch*n)

        return out


class TemporalBlock(nn.Module):
    def __init__(self, n, device):
        super(TemporalBlock, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(n, n, 1)
        self.lstm2 = nn.LSTM(n, n, 1)
        self.linear = nn.Linear(n, n)
        self.linear2 = nn.Linear(n, n)

    def forward(self, x):
        # (batch,time,node,value)
        batch, time, node, value = x.shape
        out = x.reshape(batch, time, node)

        # (batch,time,node)
        out = out.permute(1, 0, 2)

        # (time,batch,node)
        out, (a, b) = self.lstm(out)
        out, (a, b) = self.lstm2(out)
        out = out[-1, :, :]

        # (batch,node)

        out = F.relu(self.linear(out))

        return out


class TemporalComponent(nn.Module):
    def __init__(self, n, len_closeness, len_period, len_trend, device):
        super(TemporalComponent, self).__init__()

        self.num_nodes = n
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend
        self.device = device

        self.daily_block = TemporalBlock(self.num_nodes, self.device)
        self.interval_block = TemporalBlock(self.num_nodes, self.device)
        self.weekly_block = TemporalBlock(self.num_nodes, self.device)

        self.linear = nn.Linear(3*n, n)

    def forward(self, x):
        # (batch,time,node,value)
        x_interval = x[:, :self.len_closeness, :, :]
        x_daily = x[:, self.len_closeness:self.len_closeness+self.len_period, :, :]
        x_weekly = x[:, self.len_closeness+self.len_period:, :, :]

        y_interval = self.daily_block(x_interval)         # batch*n
        y_daily = self.daily_block(x_daily)     # batch*n
        y_weekly = self.weekly_block(x_weekly)   # batch*n

        out = torch.cat((y_interval, y_daily, y_weekly), 1)

        out = F.relu(self.linear(out))  # (batch*n)

        return out


class MultiSTGCnet(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # get data feature
        self.adj_mx = self.data_feature.get("adj_mx")
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.output_dim = self.data_feature.get('output_dim', 1)

        self.len_closeness = self.data_feature.get('len_closeness')
        self.len_period = self.data_feature.get("len_period")
        self.len_trend = self.data_feature.get("len_trend")

        self.input_window = config.get('input_window', 1)
        self._scaler = self.data_feature.get('scaler')
        self.device = config.get('device', torch.device('cpu'))

        # get model config
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 1)
        self.device = config.get('device', torch.device('cpu'))

        # init logger
        self._logger = getLogger()

        # define the model structure

        self.spatial_component = SpatialComponent(self.num_nodes, self.adj_mx, self.len_closeness, self.device)
        self.temporal_component = TemporalComponent(self.num_nodes,
                                                    self.len_closeness, self.len_period, self.len_trend, self.device)

        # fusion的参数
        self.Ws = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, (1, self.num_nodes)),
                                            dtype=torch.float32).to(self.device))
        self.Wt = nn.Parameter(torch.tensor(np.random.normal(0, 0.01, (1, self.num_nodes)),
                                            dtype=torch.float32).to(self.device))

        self.loss = nn.L1Loss()

        self.count = 0

    def forward(self, batch):
        # (batch,time,node,value)
        x = batch['X']

        y_spatial = self.spatial_component(x)

        y_temporal = self.temporal_component(x)

        y = torch.mul(self.Ws, y_spatial) + torch.mul(self.Wt, y_temporal)

        return y.reshape(-1, 1, self.num_nodes, 1)

    def predict(self, batch, batches_seen=None):
        return self.forward(batch)

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']                              # ground-truth value
        y_predicted = self.predict(batch, batches_seen)  # prediction results
        # denormalization the value
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # call the mask_mae loss function defined in `loss.py`
        return self.loss(y_true, y_predicted)
