import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList
import dgl
import networkx as nx
from logging import getLogger
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel


class STMGAT(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        print("========batch_g_gat_2l===========")
        heads = config.get('heads', 8)
        feat_drop = config.get('feat_drop', 0.6)
        attn_drop = config.get('attn_drop', 0.6)
        negative_slope = config.get('negative_slope', 0.2)
        receptive_field = config.get('receptive_field', 1)
        out_dim = data_feature.get('out', 12)
        residual_channels = config.get('residual_channels', 40)
        dilation_channels = config.get('dilation_channels', 40)
        skip_channels = config.get('skip_channels', 320)
        end_channels = config.get('end_channels', 640)
        kernel_size = config.get('kernel_size', 2)
        blocks = config.get('blocks', 4)
        layers = config.get('layers', 2)
        self.g = self.data_feature.get('adj_mx', 1)
        self.g = nx.from_numpy_array(self.g)
        self.g = self.g.to_directed()
        self.g = dgl.from_networkx(self.g, node_attrs=[], edge_attrs=['weight'])
        self.dropout = config.get('dropout', 0.3)
        self.blocks = blocks
        self.layers = layers
        self.run_gconv = config.get('run_gconv', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()
        self.device = config.get('device', torch.device('cpu'))
        self.g = self.g.to(self.device)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.start_conv = nn.Conv2d(in_channels=1,  # hard code to avoid errors
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.cat_feature_conv = nn.Conv2d(in_channels=1,
                                          out_channels=residual_channels,
                                          kernel_size=(1, 1))

        depth = list(range(blocks * layers))
        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])

        self.gat_layers = nn.ModuleList()
        self.gat_layers1 = nn.ModuleList()
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        # time_len change: 12/10/9/7/6/4/3/1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1  # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                # batch, channel, height, width
                # N,C,H,W
                # d = (d - kennel_size + 2 * padding) / stride + 1
                # H_out = [H_in + 2*padding[0] - dilation[0]*(kernal_size[0]-1)-1]/stride[0] + 1
                # W_out = [W_in + 2*padding[1] - dilation[1]*(kernal_size[1]-1)-1]/stride[1] + 1

                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gat_layers.append(GATConv(
                    dilation_channels*(14 - receptive_field),
                    dilation_channels*(14 - receptive_field),
                    heads, feat_drop, attn_drop, negative_slope,
                    residual=False, activation=F.elu))
                self.gat_layers1.append(GATConv(
                    dilation_channels * (14 - receptive_field),
                    dilation_channels * (14 - receptive_field),
                    heads, feat_drop, attn_drop, negative_slope,
                    residual=False, activation=F.elu))

        self.receptive_field = receptive_field
        self.end_conv_1 = Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d(end_channels, out_dim, (1, 1), bias=True)

    def forward(self, batch):
        x = batch['X']
        x = x.permute(0, 3, 2, 1)
        in_len = x.size(3)
        # Input shape is (bs, features, n_nodes, n_timesteps)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        x1 = self.start_conv(x[:, [0]])
        x2 = F.leaky_relu(self.cat_feature_conv(x[:, [0]]))
        # batch, channel, height, width
        x = x1 + x2
        skip = 0

        # STGAT layers
        for i in range(self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate
            # print("175residual:", residual.shape)
            # filter:[64, 40, 207, 12/10/9/7/6/4/3/1]

            # parametrized skip connection
            s = self.skip_convs[i](x)  # [64, 320, 207, 12/10/9]

            if i > 0:
                skip = skip[:, :, :, -s.size(3):]
            else:
                skip = 0
            skip = s + skip
            # print("183skip:", skip.shape)  # [64, 320, 207, 12/10/9]
            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            # graph conv and mix
            if self.run_gconv:
                [batch_size, fea_size, num_of_vertices, step_size] = x.size()
                # print("157 x:", x.shape)
                batched_g = dgl.batch(batch_size * [self.g])
                h = x.permute(0, 2, 1, 3).reshape(batch_size*num_of_vertices, fea_size*step_size)
                h = self.gat_layers[i](batched_g, h).mean(1)
                h = self.gat_layers1[i](batched_g, h).mean(1)

                # print("164 h:", h.shape)
                gc = h.reshape(batch_size, num_of_vertices, fea_size, -1)
                # print("166 gc:", gc.shape)
                graph_out = gc.permute(0, 2, 1, 3)
                x = x + graph_out
            else:
                x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]  # TODO(SS): Mean/Max Pool?
            # print("267 x_:", x.shape)  # [64, 40, 207, 12]
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # downsample to (bs, seq_length, 207, nfeatures)
        return x

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        res = loss.masked_mse_torch(y_predicted, y_true)
        return res

