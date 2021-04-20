from logging import getLogger
import random
import math
import torch
from torch import nn
import torch.nn.functional as F
from trafficdl.model import loss
from trafficdl.model.abstract_traffic_state_model import AbstractTrafficStateModel
import numpy as np
import scipy.sparse as sp
import networkx as nx

from collections import defaultdict


# math_utils
class math_utils:
    def __init__(self):
        super()

    def directed_adj(self):
        acc = [-1, 36, 49, 61, 79, 105, 116, 151, 162, 200]

        base = np.identity(201, dtype=bool)

        for i in range(0, 201):
            if i not in acc:
                base[i][i + 1] = True

        base[36][37] = True
        base[36][80] = True
        base[49][50] = True
        base[116][50] = True
        base[105][106] = True
        base[105][117] = True
        base[61][62] = True
        base[61][152] = True
        base[162][163] = True
        base[151][163] = True
        base[200][0] = True
        base[79][0] = True

        both = np.logical_or(base, base.transpose())

        return base.astype(int), base.transpose().astype(int), both.astype(int)

    def adjacency_matrix(self, n):
        S = np.zeros((n, n))
        for i in range(len(S)):
            if i == 0:
                S[0][1] = 1
                S[0][n - 1] = 1
            elif i == (n - 1):
                S[n - 1][n - 2] = 1
                S[n - 1][0] = 1
            else:
                S[i][i - 1] = 1
                S[i][i + 1] = 1
        return S

    def normalized_laplacian(self, adj):
        D = 1 / adj.sum(axis=1)
        D = np.diag(D)
        return adj + D

    def calculate_normalized_laplacian(self, adj):
        adj = sp.coo_matrix(adj)
        d = np.array(adj.sum(axis=1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
        return normalized_laplacian

    def calculate_adjacency_k(self, L, k):
        dim = L.shape[0]
        if k > 0:
            output = L
            for i in range(k - 1):
                output = np.matmul(output, L)
            out = np.sign(output)
        elif k == 0:
            out = np.eye(dim)
        return out

    def A_k(self, L, k):
        output = L
        for i in range(k - 1):
            output = torch.dot(output, L)
        return np.sign(output)

    def compute_threshold(self, batch, k=4):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return torch.tensor(k / (k + torch.exp(batch / k)), dtype=torch.float32)

    def rmse(self, y_true, y_pred):
        return torch.sqrt(torch.mean(torch.square(y_pred - y_true)[-1]))

    def RMSE(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    def MAPE(self, y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true))

    def MAE(self, y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    def Speed_k_correlation(self, ground_truth, weights, k):
        nb_samples = ground_truth.shape[0]
        time_steps = ground_truth.shape[1]
        N = ground_truth.shape[2]

        speed_vector = [i for i in range(115, -1, -5)]
        print(speed_vector)
        counts = []
        f_value = np.zeros(len(speed_vector))
        importances = np.zeros((len(speed_vector), 2 * k + 1))
        statistics = []
        for i in range(len(speed_vector)):
            statistics.append([])

        for i in range(len(speed_vector)):
            ground_truth[ground_truth >= speed_vector[i]] = -i - 1
            counts.append(np.count_nonzero(ground_truth == -i - 1))
        counts = np.array(counts)
        print(counts)

        W_head = weights[..., :k]
        W_end = weights[..., -k:]
        kernels = np.concatenate([W_end, weights, W_head], axis=-1)

        mask = np.array([i for i in range(-k, k + 1)])

        for i in range(nb_samples - 1):
            for j in range(time_steps):
                for l in range(N):
                    ref = ground_truth[i + 1][j][l]
                    f = 0.
                    for p in range(-k, k + 1):
                        f += mask[p + k] * kernels[i][j][l][l + p + k]
                        importances[int(-ref - 1)][p + k] += kernels[i][j][l][l + p + k]
                        # f += mask[p+k]*kernels[i][j][l][l+p]
                        # importances[int(-ref-1)][p+k] += kernels[i][j][l][l+p]
                    f_value[int(-ref - 1)] += f
                    statistics[int(-ref - 1)].append(f)

        f_out = f_value / counts

        importance = importances.transpose()
        sp = []
        for i in range(2 * k + 1):
            sp.append(importance[i] / counts)

        return f_out, np.array(sp), statistics

    def STcorrelation(self, ground_truth, weights, k):
        nb_samples = ground_truth.shape[0]
        time_steps = ground_truth.shape[1]
        N = ground_truth.shape[2]

        speed_vector = [i for i in range(115, -1, -5)]
        print(speed_vector)

        counts = np.zeros((N, len(speed_vector)))
        f_value = np.zeros((N, len(speed_vector)))
        J = np.zeros((N, len(speed_vector), 2 * k + 1))

        for i in range(len(speed_vector)):
            ground_truth[ground_truth >= speed_vector[i]] = -i - 1

        W_head = weights[..., :k]
        W_end = weights[..., -k:]
        kernels = np.concatenate([W_end, weights, W_head], axis=-1)
        print(kernels.shape)

        mask = np.array([i for i in range(-k, k + 1)])

        for i in range(nb_samples - 1):
            for j in range(time_steps):
                filters = kernels[i][j]
                for l in range(N):
                    ref = ground_truth[i + 1][j][l]
                    counts[l][int(-ref - 1)] += 1
                    f = 0
                    for p in range(-k, k + 1):
                        f += mask[p + k] * filters[l][l + p + k]
                        J[l][int(-ref - 1)][p + k] += filters[l][l + p + k]
                    f_value[l][int(-ref - 1)] += f

        f_out = f_value / counts
        sp = []
        for i in range(2 * k + 1):
            sp.append(J[..., i] / counts)
        st_out = np.array(sp)
        print(st_out.shape)

        return f_out, st_out

    def Compare(self, ground_truth, weights, k):
        nb_samples = ground_truth.shape[0]
        N = ground_truth.shape[1]

        f_value = np.zeros((nb_samples, N))

        W_head = weights[..., :k]
        W_end = weights[..., -k:]
        kernels = np.concatenate([W_end, weights, W_head], axis=-1)
        print(kernels.shape)

        mask = np.array([i for i in range(-k, k + 1)])

        for i in range(nb_samples):
            filters = kernels[i]
            for l in range(N):
                f1 = 0
                for p in range(-k, k + 1):
                    f1 += mask[p + k] * filters[l][l + p + k]
                f_value[i][l] = f1
        print(f_value.shape)

        return f_value

    def Track(self, data, mask, Ad, Au, A, k, node):
        nb_samples, time_steps, N = data.shape

        G = nx.from_numpy_matrix(A)
        dis = dict(nx.all_pairs_shortest_path_length(G))

        # Ak = LA.matrix_power(A, k)
        Ak = torch.pow(A, k)

        lg = np.count_nonzero(mask[0][0][node])
        print(lg, 'nodes')

        speed = np.zeros((nb_samples, time_steps, lg))
        weights = np.zeros((nb_samples, time_steps, lg))

        index = []

        for i in range(N):
            if Ak[node][i] != 0:
                index.append(i)

        print(index)

        for i in range(nb_samples):
            for j in range(time_steps):
                m = mask[i][j][node]  # N
                raw = data[i][j]

                v = []
                w = []

                for r in range(N):
                    if m[r] != 0:
                        v.append(raw[r])
                        w.append(m[r])

                speed[i][j] = np.array(v)
                weights[i][j] = np.array(w)

        return speed, weights

    def z_score(self, x, mean, std):
        return (x - mean) / std

    def z_inverse(self, x, mean, std):
        return x * std + mean


# layers_keras
dim = 208  # number of nodes
# construct adjacency matrix and scaled Laplacian for a circle graph with dim nodes
A = math_utils.adjacency_matrix(dim)  # for ROTnet
# Ad, Au, A = directed_adj() # for AMSnet
scaled_laplacian = math_utils.normalized_laplacian(A)


class LocallyConnectedGC(nn.Module):
    """
    Locally-connected graph convolutional layer
        Arguments:
            k: receptive field, k-hop neighbors
            scaled_lap: scaled Laplacian matrix
    """

    def __init__(self, k,
                 scaled_lap=scaled_laplacian,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(LocallyConnectedGC, self).__init__(**kwargs)

        self.k = k
        self.scaled_lap = scaled_lap

        if activation == 'tanh':
            self.activation = nn.functional.tanh
        elif activation == 'softmax':
            self.activation = nn.functional.Softplus

        '''self.activation = activations.get(activation)               # nn.functional.Tanh or Softplus'''
        self.use_bias = use_bias
        '''self.kernel_initializer = initializers.get(kernel_initializer) #torch.nn.init.uniform(tensor, a=0, b=1)
        self.bias_initializer = initializers.get(bias_initializer)   # torch.Tensor(0)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)'''
        self.supports_masking = True
        self.supports = []
        # k-hop adjacency matrix
        # torch.sparse.to_dense()
        #

        # S = K.constant(K.to_dense(math_utils.calculate_adjacency_k(self.scaled_lap, self.k))) # 常数张量，稀疏张量转换为稠密张量
        S = torch.nn.init.constant(torch.Tensor.to_dense(math_utils.calculate_adjacency_k(self.scaled_lap, self.k)))
        self.supports.append(S)

    def compute_output_shape(self, input_shapes):
        return input_shapes

    def build(self, input_shapes):
        nb_nodes = input_shapes[1]
        feature_dim = input_shapes[-1]

        self.kernel = self.add_weight(shape=(nb_nodes, nb_nodes),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(feature_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs
        W = self.supports[0] * self.kernel
        X = torch.dot(torch.transpose(features, 1, 2), W)  # dot 点乘，重新排列张量的轴
        outputs = torch.transpose(X, 1, 2)
        if self.use_bias:
            outputs += self.bias
        return self.activation(outputs)


class Linear(nn.Module):
    """
    Node-wise dense layer wih given output dimension
        Arguments:
            units: dimension of output
    """

    def __init__(self, units=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Linear, self).__init__(**kwargs)
        self.units = units

        # 根据传入激活函数名称选择对应的激活函数
        # self.activation = activations.get(activation)
        if activation == 'tanh':
            self.activation = nn.functional.tanh
        elif activation == 'hard_sigmoid':
            self.activation = nn.functional.hardsigmoid

        self.use_bias = use_bias
        '''
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        '''
        self.supports_masking = True

    def compute_output_shape(self, input_shapes):
        output_shape = (None, input_shapes[1], self.units)
        return output_shape

    def build(self, input_shapes):
        input_dim = input_shapes[-1]
        self.kernel_lin = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel_lin',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        outputs = torch.dot(inputs, self.kernel_lin)
        if self.use_bias:
            outputs += self.bias
        return self.activation(outputs)


class Scheduled(nn.Module):
    """
    Scheduled sampling layer, input = [ground_truth, previous_prediction]
        Arguments:
            coin: probability to use previous prediction, default = 0
        Methods:
            reset_coin: to change the probability
    """

    def __init__(self, coin=0., **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Scheduled, self).__init__(**kwargs)
        self.coin = coin

    def compute_output_shape(self, input_shapes):
        return input_shapes[-1]

    def build(self, input_shapes):
        self.built = True

    def reset_coin(self, s):
        self.coin = s

    def call(self, inputs, mask=None):
        rand = random.uniform(0, 1)
        if rand > self.coin:
            # outputs = K.permute_dimensions(inputs[0], (0, 2, 1))
            outputs = torch.transpose(inputs[0], 1, 2)
        else:
            outputs = inputs[1]
        return outputs

    def get_config(self):
        config = {'coin': self.coin}

        base_config = super(Scheduled, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DynamicGCf(nn.Module):
    """
    Dynamic graph convolutional module, Eq-(5)/(DGC) in the paper
        Arguments:
            k: receptive field
            units: output dimension
            normalization: use softmax normalization or not, defaut = None
            attn_heads=1,
            attn_heads_reduction='concat',  # {'concat', 'average'}
            scaled_lap = scaled_laplacian,
    """

    def __init__(self,
                 k,
                 units=1,
                 normalization=False,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 scaled_lap=scaled_laplacian,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.k = k
        self.units = units
        self.normalization = normalization
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.scaled_lap = scaled_lap
        # self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        # self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)

        # self.kernel_regularizer = regularizers.get(kernel_regularizer)
        # self.bias_regularizer = regularizers.get(bias_regularizer)
        # self.activity_regularizer = regularizers.get(activity_regularizer)

        # self.kernel_constraint = constraints.get(kernel_constraint)
        # self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []  # Layer kernels for attention heads
        self.parameters = []
        self.fc = []
        self.biases = []  # Layer biases for attention heads

        self.supports = []

        # s = K.constant(K.to_dense(math_utils.calculate_adjacency_k(self.scaled_lap, self.k)))
        s = torch.nn.init.constant(torch.Tensor.to_dense(math_utils.calculate_adjacency_k(self.scaled_lap, self.k)))

        self.supports.append(s)

        super(DynamicGCf, self).__init__(**kwargs)

    def build(self, input_shape):
        F = input_shape[-1]
        N = input_shape[-2]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, N),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name='kernel_{}'.format(head))

            self.kernels.append(kernel)
            # Layer parameters
            parameter = self.add_weight(shape=(N, N),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        name='parameter_{}'.format(head))
            self.parameters.append(parameter)

            # # Layer bias
            if self.use_bias:
                bias1 = self.add_weight(shape=(N,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias1_{}'.format(head))

                bias2 = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name='bias2_{}'.format(head))
                self.biases.append([bias1, bias2])

        self.built = True

    def call(self, inputs):
        X = inputs  # Node features (N x F)
        speed = inputs[:, :, :1]
        A = self.supports[0]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # [1]  # W in the paper (F x N)
            W = self.parameters[head]
            bias_N = self.biases[head][0]
            bias_units = self.biases[head][1]

            feature = torch.dot(torch.transpose(X, 1, 2), A * W)
            dense = torch.dot(torch.transpose(feature, 1, 2), kernel)
            # dense = K.dot(dense, kernel)
            if self.use_bias:
                dense = dense + bias_N
                # dense = K.bias_add(dense, bias_N)

            if self.normalization:
                mask = dense + -10e15 * (1.0 - A)
                mask = torch.nn.functional.softmax(mask)
            else:
                mask = dense * A

            # Linear combination with neighbors' features
            # node_features = K.batch_dot(mask, feature)
            mask = torch.unsqueeze(mask, 1)
            speed = torch.unsqueeze(speed, 1)
            node_features = torch.matmul(mask, speed)  # (N x F)

            if self.use_bias:
                node_features = node_features + bias_units
                # node_features = K.bias_add(node_features, bias_units)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output = torch.cat(outputs)
            # output = K.concatenate(outputs)  # (N x KF')
        else:
            output = torch.mean(torch.stack(outputs), dim=1)  # N x F')
            # output = K.mean(K.stack(outputs), axis=1)  # N x F')

        output = self.activation(output)
        return output

        # for model interpretation
        # return K.concatenate([output, mask], axis=-1)

    def compute_output_shape(self, input_shapes):
        if self.attn_heads_reduction == 'concat':
            output_shape = (input_shapes[0], input_shapes[1], self.units * self.attn_heads)

            # for model interpretation
            # output_shape = (input_shapes[0], input_shapes[1], self.units*self.attn_heads + input_shapes[1])
        else:
            output_shape = (input_shapes[0], input_shapes[1], self.units)
        return output_shape


class DGCRNNCell(nn.Module):
    """
    RNN cell with GRU structure and spatial attention gates
    Eq-(DGGRU) in the paper
        Arguments:
            k: receptive field - 1
            dgc_mode: spatial attention module,
                {'dgc','gan','lc'}
    """

    def __init__(self, k,
                 dgc_mode='hybrid',
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DGCRNNCell, self).__init__(**kwargs)
        self.k = k
        self.dgc_mode = dgc_mode
        self.state_size = dim
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.supports_masking = True

    def build(self, input_shape):
        # input_shape = (bs, nb_nodes)
        shapes = (input_shape[0], input_shape[1], 2)
        inner = (input_shape[0], input_shape[1], 2 * self.k + 1)
        shapes_1 = (input_shape[0], input_shape[1], 1)

        if self.dgc_mode == 'hybrid':
            self.dgc_r = LocallyConnectedGC(2)  # , depthwise=False
            self.dgc_r.build(shapes)
            wr_1 = self.dgc_r.trainable_weights

            self.lin_r = Linear(1, activation=self.recurrent_activation)
            self.lin_r.build(shapes)
            wr_2 = self.lin_r.trainable_weights

            self.dgc_u = LocallyConnectedGC(2)  # , depthwise=False
            self.dgc_u.build(shapes)
            wu_1 = self.dgc_u.trainable_weights

            self.lin_u = Linear(1, activation=self.recurrent_activation)
            self.lin_u.build(shapes)
            wu_2 = self.lin_u.trainable_weights

            self.dgc_c = LocallyConnectedGC(2)  # , depthwise=False
            self.dgc_c.build(shapes)
            wc_1 = self.dgc_c.trainable_weights

            self.lin_c = Linear(1, activation=self.activation)
            self.lin_c.build(shapes)
            wc_2 = self.lin_c.trainable_weights

            self.core = DynamicGCf(k=self.k, units=1, normalization=True, activation=None)
            self.core.build(shapes)
            w_1 = self.core.trainable_weights

            w = wr_1 + wr_2 + wu_1 + wu_2 + wc_1 + wc_2 + w_1

        self._trainable_weights = w
        self.built = True

    def call(self, inputs, states):

        feature_ru1 = torch.cat([torch.unsqueeze(inputs), torch.unsqueeze(states[0])])
        # feature_ru1 = K.concatenate([K.expand_dims(inputs), K.expand_dims(states[0])])  # (bs, nb_nodes, 2)
        p = self.core(feature_ru1)

        feature_ru = torch.cat([p, torch.unsqueeze(states[0])])
        # feature_ru = K.concatenate([p, K.expand_dims(states[0])])

        r = self.dgc_r(feature_ru)  # (bs, nb_nodes, 2)
        r = self.lin_r(r)  # (bs, nb_nodes, 1)

        u = self.dgc_u(feature_ru)  # (bs, nb_nodes, 2)
        u = self.lin_u(u)  # (bs, nb_nodes, 1)

        s = r * torch.unsqueeze(states[0])  # (bs, nb_nodes, 1)
        # s = r * K.expand_dims(states[0])  # (bs, nb_nodes, 1)
        feature_c = torch.cat([torch.unsqueeze(inputs), s])  # (bs, nb_nodes, 2)
        # feature_c = torch.cat([K.expand_dims(inputs), s])  # (bs, nb_nodes, 2)

        c = self.dgc_c(feature_c)  # (bs, nb_nodes, 2)
        c = self.lin_c(c)  # (bs, nb_nodes, 1)

        H = u * torch.unsqueeze(states[0]) + (1 - u) * c
        # H = u * K.expand_dims(states[0]) + (1 - u) * c

        return torch.squeeze(p, dim=-1), [torch.squeeze(H, dim=-1)]
        # return K.squeeze(p, axis=-1), [K.squeeze(H, axis=-1)]


class ScheduledSampling('''Callback'''):
    '''
    Custom Callbacks to use teacher forcing and schedule sampling
        strategy in training.
        see `https://arxiv.org/pdf/1506.03099.pdf`
    # Argument:
        k: Interger, the coefficient to control convergence speed
        model: the keras model to apply
    '''

    def __init__(self, k, **kwargs):
        self.k = k
        super(ScheduledSampling, self).__init__(**kwargs)

    def on_train_epoch_begin(self, epoch, logs=None):
        prob_use_gt = self.k / (self.k + torch.exp(epoch / self.k))
        for layer in self.model.layers:
            if 'reset_coin' in dir(layer):
                layer.reset_coin(prob_use_gt)

    def on_test_batch_begin(self, batch, logs=None):
        for layer in self.model.layers:
            if 'reset_coin' in dir(layer):
                layer.reset_coin(0.)

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if 'reset_coin' in dir(layer):
                layer.reset_coin(0.)


# model_keras.py
def unstack(x):
    x = torch.unbind(x, dim=1)
    # x = tf.unstack(x, axis=1)
    return x


def stack(x):
    y = torch.stack(x, dim=1)
    # y = K.stack(x, axis=1)
    return torch.squeeze(y, dim=2)
    # return K.squeeze(y, axis=2)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def Lambda(self, x):
        return self.lambd(x)


'''
def create_embed_model(obs_timesteps=10, pred_timesteps=3, nb_nodes=208, k=1, dgc_mode='hybrid', inner_act=None):
    #encoder = RNN(DGCRNNCell(k, dgc_mode=dgc_mode), return_state=True)
    #decoder = RNN(DGCRNNCell(k, dgc_mode=dgc_mode), return_sequences=True, return_state=True)
    encoder = torch.nn.RNN(DGCRNNCell(k, dgc_mode=dgc_mode), return_state=True)
    decoder = torch.nn.RNN(DGCRNNCell(k, dgc_mode=dgc_mode), return_sequences=True, return_state=True)

    unstack_k = LambdaLayer.Lambda(unstack)
    choice = Scheduled()

    input_obs = torch.randn(size=(obs_timesteps, nb_nodes, 1))
    # input_obs = Input(shape=(obs_timesteps, nb_nodes, 1)) ????????????????
    input_gt = torch.randn(size=(pred_timesteps, nb_nodes, 1))  # (None, T, N, 1)
    # input_gt = Input(shape=(pred_timesteps, nb_nodes, 1))  # (None, T, N, 1)
    encoder_inputs = LambdaLayer.Lambda(lambda x: torch.squeeze(x, dim=-1))(input_obs)  # (None, T, N)
    # encoder_inputs = Lambda(lambda x: K.squeeze(x, axis=-1))(input_obs)  # (None, T, N)

    encoder_outputs, state_h = encoder(encoder_inputs)

    unstacked = unstack_k(input_gt)  # [(None, N, 1) x T] list

    initial = unstacked[0]  # (None, N, 1)
    decoder_inputs = LambdaLayer.Lambda(lambda x: torch.permute_dimensions(x, 1, 2))(initial)  # (None, 1, N)
    decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
    state_h = state_h_new

    # prediction part
    prediction = []
    decoded_results = decoder_outputs_new
    prediction.append(decoded_results)

    if pred_timesteps > 1:
        for i in range(1, pred_timesteps):
            decoder_inputs = choice([prediction[-1], unstacked[i]])  # (None, 208, 1)
            decoder_inputs = LambdaLayer.Lambda(lambda x: torch.permute_dimensions(x, 1, 2))(decoder_inputs)  # (None, 1, 208)
            decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
            state_h = state_h_new
            decoded_results = decoder_outputs_new
            prediction.append(decoded_results)

    outputs = LambdaLayer.Lambda(stack)(prediction)
    model = Model([input_obs, input_gt], outputs)


    return model
'''


# DGFN
class DGFN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        # 1.初始化父类
        super().__init__(config, data_feature)

        # 2.从data_feature(TrafficStatePointDataset)中获取需要的信息
        self._scaler = self.data_feature.get('scaler')  # 归一化方法
        self.adj_mx = self.data_feature.get('adj_mx')  # 邻接矩阵
        self.num_nodes = self.data_feature.get('adj_mx')  # 点的个数
        self.feature_dim = self.data_feature.get('feature_dim')  # 输入数据的维度
        self.output_dim = self.data_feature.get('output_dim')  # 模型输出的维度

        # 3.初始化log用于必要的输出
        self._logger = getLogger()

        # 4.初始化device
        self.device = config.get('device', torch.device('cpu'))

        # 5.Build network
        self.encoder = torch.nn.RNN()
        self.decoder = torch.nn.RNN()
        self.input_obs = torch.randn()
        self.input_gt = torch.randn()
        self.encoder_inputs = LambdaLayer.Lambda(lambda x: torch.squeeze(x, dim=-1))(self.input_obs)
        self.encoder_outputs, state_h = self.encoder(self.encoder_inputs)
        self.unstacked = self.unstack_k(self.input_gt)
        self.initial = self.unstacked[0]  # (None, N, 1)
        self.decoder_inputs = LambdaLayer.Lambda(lambda x: torch.transpose(x, 1, 2))(self.initial)
        self.decoder_outputs_new, self.state_h_new = self.decoder(self.decoder_inputs, initial_state=state_h)
        self.state_h = self.state_h_new

        self.prediction = []
        self.decoded_results = self.decoder_outputs_new
        self.prediction.append(self.decoded_results)
        self.pred_timesteps = 3
        self.choice = Scheduled()

        if self.pred_timesteps > 1:
            for i in range(1, self.pred_timesteps):
                self.decoder_inputs = self.choice([self.prediction[-1], self.unstacked[i]])  # (None, 208, 1)
                self.decoder_inputs = LambdaLayer.Lambda(lambda x: torch.transpose(x, 1, 2))(
                    self.decoder_inputs)  # (None, 1, 208)
                self.decoder_outputs_new, self.state_h_new = self.decoder(self.decoder_inputs, initial_state=state_h)
                self.state_h = self.state_h_new
                self.decoded_results = self.decoder_outputs_new
                self.prediction.append(self.decoded_results)

        self.outputs = LambdaLayer.Lambda(stack)(self.prediction)

    def forward(self, batch):
        # 1.取数据，模型输入的特征维度应该等于self.feature_dim
        x = batch['X']

        # 2.根据输入数据计算模型的输出结果，模型输出的特征维度应该等于self.output_dim
        k = 1
        dgc_mode = 'hybrid'
        obs_timesteps = 10
        pred_timesteps = 3
        nb_nodes = 208
        encoder = torch.nn.RNN(DGCRNNCell(k, dgc_mode=dgc_mode), return_state=True)
        decoder = torch.nn.RNN(DGCRNNCell(k, dgc_mode=dgc_mode), return_sequences=True, return_state=True)

        unstack_k = LambdaLayer.Lambda(unstack)
        choice = Scheduled()

        input_obs = torch.randn(size=(obs_timesteps, nb_nodes, 1))
        input_gt = torch.randn(size=(pred_timesteps, nb_nodes, 1))  # (None, T, N, 1)
        encoder_inputs = LambdaLayer.Lambda(lambda x: torch.squeeze(x, dim=-1))(input_obs)  # (None, T, N)

        encoder_outputs, state_h = encoder(encoder_inputs)

        unstacked = unstack_k(input_gt)  # [(None, N, 1) x T] list

        initial = unstacked[0]  # (None, N, 1)
        decoder_inputs = LambdaLayer.Lambda(lambda x: torch.transpose(x, 1, 2))(initial)  # (None, 1, N)
        decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
        state_h = state_h_new

        # prediction part
        prediction = []
        decoded_results = decoder_outputs_new
        prediction.append(decoded_results)

        if pred_timesteps > 1:
            for i in range(1, pred_timesteps):
                decoder_inputs = choice([prediction[-1], unstacked[i]])  # (None, 208, 1)
                decoder_inputs = LambdaLayer.Lambda(lambda x: torch.transpose(x, 1, 2))(decoder_inputs)
                decoder_outputs_new, state_h_new = decoder(decoder_inputs, initial_state=state_h)
                state_h = state_h_new
                decoded_results = decoder_outputs_new
                prediction.append(decoded_results)

        outputs = LambdaLayer.Lambda(stack)(prediction)

        return outputs

    def calculate_loss(self, batch):
        # 1.取出真值
        y_true = batch['y']

        # 2.取出预测值
        y_predicted = self.predict(batch)

        # 3.使用self._scaler将进行了归一化的真值和预测值进行反向归一化（必须）
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # 4.调用loss函数计算真值和预测值的误差
        res = (abs(y_true - y_predicted) / y_true).sum()

        # 5.返回loss的结果
        return res

    def predict(self, batch):
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        output_length = y.shape[1]
        y_preds = []
        x_ = x.clone()  # copy!!
        for i in range(output_length):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1(output_length), num_nodes, 1(feature_dim))
            y_preds.append(y_.clone())
            if y_.shape[3] < x_.shape[3]:  # y_的feature_dim可能小于x_的
                y_ = torch.cat([y_, y[:, i:i + 1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # concat at time_length, y_preds.shape=y.shape
        return y_preds
