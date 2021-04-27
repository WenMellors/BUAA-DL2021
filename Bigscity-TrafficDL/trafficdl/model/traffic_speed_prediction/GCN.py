import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from trafficdl.lib.utils import scaled_Laplacian, cheb_polynomial


class GCN(nn.Module):
    '''
    K-order chebyshev graph convolution
    计算khop的GCN
    '''

    def __init__(self, adj, dim_in, dim_out, order_K, device, in_drop=0.0, gcn_drop=0.0, residual=False):
        '''
        :param adj:邻接矩阵
        :param K: int,num of aggregated neighbors
        :param dim_in: int, num of channels in the input sequence
        :param dim_out: int, num of channels in the output sequence
        '''
        super(GCN, self).__init__()
        self.DEVICE = device
        self.order_K = order_K
        self.adj = adj
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(dim_in, dim_out))
             for _ in range(order_K)])
        self.weights = nn.Parameter(torch.FloatTensor(size=(dim_out, dim_out)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(dim_out,)))
        self._in_drop = in_drop
        self._gcn_drop = gcn_drop
        self._residual = residual
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x, state=None, M=None):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size,N, dim_in)
        :return: (batch_size,N, dim_out)
        '''
        batch_size, num_of_vertices, in_channels = x.shape
        output = torch.zeros(batch_size, num_of_vertices, self.dim_out).to(self.DEVICE)  # (batch_size,N, dim_out)
        L_tilde = scaled_Laplacian(self.adj)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor) for i in
                            cheb_polynomial(L_tilde, self.order_K)]
        if state is not None:
            s = torch.einsum('ij,jkm->ikm', M, state.permute(1, 0, 2)).permute(1, 0, 2)
            x = torch.cat((x, s), dim=-1)
        x0 = x
        if self._in_drop != 0:
            x = torch.dropout(x, 1.0 - self._in_drop, train=True)
        # k-order展开
        for k in range(self.order_K):
            # chebyshev多项式
            output = output + x.permute(0, 2, 1).matmul(cheb_polynomials[k].to(self.DEVICE)).permute(0, 2, 1).matmul(self.Theta[k])
        output = torch.matmul(output, self.weights)
        output = output + self.biases
        res = F.relu(output)
        if self._gcn_drop != 0.0:
            res = torch.dropout(res, 1.0 - self._gcn_drop, train=True)
        if self._residual:
            x0 = self.linear(x0)
            res = res + x0
        return res  # (batch_size,N, dim_out)