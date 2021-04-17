from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SANN(nn.Module):
    def __init__(self, n_inp, n_out, t_inp, t_out, n_points, past_t, hidden_dim, dropout):
        super(SANN, self).__init__()
        # Variables
        self.n_inp = n_inp
        self.n_out = n_out
        self.t_inp = t_inp
        self.t_out = t_out
        self.n_points = n_points
        self.past_t = past_t
        self.hidden_dim = hidden_dim
        # Convolutional layer
        self.conv_block = AgnosticConvBlock(n_inp, n_points, past_t, hidden_dim, num_conv=1)
        self.convT = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, n_points))
        # Regressor layer
        self.regressor = ConvRegBlock(t_inp, t_out, n_points, hidden_dim)
        # Dropout
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        N, C, T, S = x.size()
        # Padding
        xp = F.pad(x, pad=(0, 0, self.past_t - 1, 0))
        # NxCxTxS ---> NxHxTx1
        out = self.conv_block(xp)
        out = out.view(N, self.hidden_dim, T, 1)
        # NxHxTx1 ---> NxHxTxS
        out = self.convT(out)
        # 2D dropout
        out = self.drop(out)
        # NxHxTxS ---> NxC'xT'xS
        out = self.regressor(out.view(N, -1, S))
        return out.view(N, self.n_out, self.t_out, self.n_points)


class AgnosticConvBlock(nn.Module):
    def __init__(self, n_inp, n_points, past_t, hidden_dim, num_conv):
        super(AgnosticConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=n_inp, out_channels=hidden_dim, kernel_size=(past_t, n_points), bias=True),
                  nn.BatchNorm2d(num_features=hidden_dim, affine=True, track_running_stats=True),
                  nn.ReLU()]
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class ConvRegBlock(nn.Module):
    def __init__(self, t_inp, t_out, n_points, hidden_dim):
        super(ConvRegBlock, self).__init__()
        layers = [nn.Conv1d(in_channels=hidden_dim * t_inp, out_channels=t_out, kernel_size=1, bias=True),
                  nn.BatchNorm1d(num_features=t_out, affine=True, track_running_stats=True)]
        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)


class ATDM(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # get data feature
        self.adj_mx = self.data_feature.get('adj_mx')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        # get model config
        self.hidden_size = config.get('hidden_size', 64)
        self.device = config.get('device', torch.device('cpu'))
        self.input_window = config.get('input_window', 12)
        self.output_window = config.get('output_window', 3)
        self.past_t = config.get('past_t', 3)
        self.dropout = config.get('dropout', 0.2)
        # init logger
        self._logger = getLogger()
        # define the model structure
        self.sann = SANN(self.feature_dim, self.output_dim, self.input_window, self.output_window, self.num_nodes,
                         self.past_t, self.hidden_size,
                         self.dropout)

    def forward(self, x):
        # bz x feature_dim(1) x input_window(12) x num_nodes(207) ->
        # bz x output_dim(1) x output_window(3) x num_nodes(207)
        return self.sann(x)

    def calculate_loss(self, batch):
        output_y = self.predict(batch)  # bz x T' x S x 1
        input_y = batch['y']
        # input_y = [torch.from_numpy(array.astype(np.float32)) for array in input_y]
        # input_y = torch.stack(input_y)  # bz x T' x S x 1

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output_y, input_y)
        return loss

    def predict(self, batch):
        input_x = batch['X'].permute(0, 3, 1, 2)  # list[ndarray]

        # input_x = [torch.from_numpy(array.astype(np.float32)) for array in
        #           input_x]  # !! error merges if float64(double), bz x T x S x 1
        # input_x = torch.stack(input_x).permute(0, 3, 1, 2)  # bz x 1 x T x S
        output_y = self.forward(input_x)  # bz x 1 x T' x S
        return output_y.permute(0, 2, 3, 1)  # bz x T' x S x 1
