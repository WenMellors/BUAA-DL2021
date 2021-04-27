import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
from trafficdl.model import loss
from scipy.sparse.linalg import eigs

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

def scaled_laplacian(weight):
    assert weight.shape[0] == weight.shape[1]
    diag = np.diag(np.sum(weight, axis=1))
    lap = diag - weight
    lambda_max = eigs(lap, k=1, which='LR')[0].real
    return (2 * lap) / lambda_max - np.identity(weight.shape[0])

def cheb_polynomial(l_tilde, k):
    num = l_tilde.shape[0]
    cheb_polynomials = [np.identity(num), l_tilde.copy()]
    for i in range(2, k):
        cheb_polynomials.append(2 * l_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    return cheb_polynomials

class ChebConv(nn.Module):
    def __init__(self, k, cheb_polynomials, in_channels, out_channels):
        super(ChebConv, self).__init__()
        self.K = k
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(k)])
    
    def forward(self, x):
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        outputs = []
        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]  
            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  
            for k in range(self.K):
                t_k = self.cheb_polynomials[k]  
                theta_k = self.Theta[k]  
                rhs = graph_signal.permute(0, 2, 1).matmul(t_k).permute(0, 2, 1)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))

class Residual_block(nn.Module):
    def __init__(self, in_channels, k, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials):
        super(Residual_block, self).__init__()
        self.ChebConv = ChebConv(k, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        
    def forward(self, x):
        spatial_gcn = self.ChebConv(x)  # (b,N,F,T)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))
        time_conv_output = wave_net_activation(time_conv_output)
        out = nn.Dropout(0.2)(x)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  
        x_residual = (x_residual + time_conv_output).permute(0, 3, 2, 1).permute(0, 2, 3, 1)  
        return x_residual


class FullConnecLayer(nn.Module):
    def __init__(self, n, h, w, device):
        super(FullConnecLayer, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(1, n, h, w).to(device))

    def forward(self, x):
        x = x * self.weights  
        return x


class TCNSubmodule(nn.Module):
    def __init__(self, device, nb_block, in_channels, k, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, output_window, output_dim, num_of_vertices):
        super(TCNSubmodule, self).__init__()
        self.BlockList = nn.ModuleList([Residual_block(in_channels, k, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials)])
        self.BlockList.extend([Residual_block(nb_time_filter, k, nb_chev_filter, nb_time_filter, 1, cheb_polynomials) for _ in range(nb_block-1)])
        self.final_conv = nn.Conv2d(output_window, output_window, kernel_size=(1, nb_time_filter - output_dim + 1))
        self.FullConnecLayer = FullConnecLayer(output_window, num_of_vertices, output_dim, device)
        self.lstm = nn.LSTM(input_size=325, hidden_size=325, num_layers=2)
        
    def forward(self, x):
        """
        Args:
            x: (B, T_in, N_nodes, F_in)
        Returns:
            torch.tensor: (B, T_out, N_nodes, out_dim)
        """
        x = x.permute(0, 2, 3, 1)  # (B, N, F_in(feature_dim), T_in)
        for block in self.BlockList:
            x = block(x)
        output = self.final_conv(x.permute(0, 3, 1, 2))
        outp = output.squeeze()
        outp,_ = self.lstm(outp)
        output = outp.unsqueeze(3)
        output = nn.Dropout(0.2)(output)
        output = self.FullConnecLayer(output)
        output = torch.sigmoid(output)
        return output


class MSTGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.len_period = self.data_feature.get('len_period', 0)
        self.len_trend = self.data_feature.get('len_trend', 0)
        self.len_closeness = self.data_feature.get('len_closeness', 0)
        if self.len_period == 0 and self.len_trend == 0 and self.len_closeness == 0:
            raise ValueError('Num of days/weeks/hours are all zero! Set at least one of them not zero!')
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.nb_block = config.get('nb_block', 2)
        self.K = config.get('K', 3)
        self.nb_chev_filter = config.get('nb_chev_filter', 64)
        self.nb_time_filter = config.get('nb_time_filter', 64)
        adj_mx = self.data_feature.get('adj_mx')
        l_tilde = scaled_laplacian(adj_mx)
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.device)
                                 for i in cheb_polynomial(l_tilde, self.K)]
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        if self.len_closeness > 0:
            self.hours_TCN_submodule = \
                TCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_closeness // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        if self.len_period > 0:
            self.days_TCN_submodule = \
                TCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_period // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        if self.len_trend > 0:
            self.weeks_TCN_submodule = \
                TCNSubmodule(self.device, self.nb_block, self.feature_dim,
                                self.K, self.nb_chev_filter, self.nb_time_filter,
                                self.len_trend // self.output_window, self.cheb_polynomials,
                                self.output_window, self.output_dim, self.num_nodes)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        x = batch['X']  # (B, Tw+Td+Th, N_nodes, F_in)
        # print(x.shape)
        # 时间维度(第1维)上的顺序是CPT，即
        # [0, len_closeness) -- input1
        # [len_closeness, len_closeness+len_period) -- input2
        # [len_closeness+len_period, len_closeness+len_period+len_trend) -- input3
        output = 0
        if self.len_closeness > 0:
            begin_index = 0
            end_index = begin_index + self.len_closeness
            output_hours = self.hours_TCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_hours
        if self.len_period > 0:
            begin_index = self.len_closeness
            end_index = begin_index + self.len_period
            output_days = self.days_TCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_days
        if self.len_trend > 0:
            begin_index = self.len_closeness + self.len_period
            end_index = begin_index + self.len_trend
            output_weeks = self.weeks_TCN_submodule(x[:, begin_index:end_index, :, :])
            output += output_weeks
        return output  # (B, Tp, N_nodes, F_out)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return torch.mean(correntropy(y_predicted, y_true))

    def predict(self, batch):
        return self.forward(batch)
