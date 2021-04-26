import torch
import torch.nn as nn
import torch.nn.functional as F
from trafficdl.model.traffic_speed_prediction.GCN import GCN
import time


class BGCN(nn.Module):
    '''
    Multi range Gcn
    计算不同范围GCN
    '''

    def __init__(self, adj_node, adj_edge, dim_in_node, dim_out_node, dim_out_edge, M, range_K, device, in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        '''
        :param range_K: k ranges
        :param adj_node:(V,V)
        :param adj_edge:(E,E)
        :param V:number of node
        :param E:number of edge
        :param M:(V,E),M_{i,(i->j)}=M_{j,(i->j)}=1
        :param dim_in_node: int, num of channels in the input sequence
        :param dim_out_node: int, num of node channels  in the output sequence
        :param dim_out_edge: int, num of edge channels  in the output sequence
        '''
        super(BGCN, self).__init__()
        self.DEVICE = device
        self.K = range_K
        self._M = M
        GCN_khops_node = []

        for k in range(self.K):
            if k == 0:
                GCN_khops_node.append(
                    GCN(adj_node, dim_in_node, dim_out_node, k + 1, device, in_drop=in_drop, gcn_drop=gcn_drop,
                        residual=residual))
            else:
                GCN_khops_node.append(
                    GCN(adj_node, dim_out_node + dim_out_edge, dim_out_node, k + 1, device, in_drop=in_drop,
                        gcn_drop=gcn_drop,
                        residual=residual))
        self.GCN_khops_node = nn.ModuleList(GCN_khops_node)
        self.GCN_khops_edge = nn.ModuleList(
            [GCN(adj_edge, dim_out_edge, dim_out_edge, k + 1, device, in_drop=in_drop, gcn_drop=gcn_drop,
                 residual=residual) for
             k in range(self.K)])

        self.W_b = nn.Parameter(torch.FloatTensor(dim_in_node, dim_out_edge))

    def forward(self, X):
        '''
        计算k个不同范围邻居的GCN
        :param X: (batch_size,N, dim_in_node)
        :return: (K,batch_size,N, dim_out_node)
        '''
        Xs = []
        Z0 = torch.einsum('ijk,km->ijm', (torch.einsum('ij,jmk->imk', self._M.permute(1, 0), X.permute(1, 0, 2))),
                          self.W_b).permute(1, 0, 2)  # (b,E,edge_num_units)
        for k in range(self.K):
            # start=time.time()
            Z0 = self.GCN_khops_edge[k](Z0)  # (b,E,edge_num_units)
            # print(time.time() - start)
            start = time.time()
            if k == 0:
                X = self.GCN_khops_node[k](X)  # (b,V,node_num_units)
            else:
                X = self.GCN_khops_node[k](X, Z0, self._M)  # (b,V,node_num_units)
            # print(time.time() - start)
            Xs.append(X)
        Xs = torch.stack(Xs)  # (K,b,V,node_num_units)
        return Xs


class MRABGCN(nn.Module):
    '''
    计算不同范围邻居的GCN输出的权重
    '''

    def __init__(self, adj_node, adj_edge, dim_in_node, dim_out_node, dim_out_edge, M, range_K, dim_out, device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        super(MRABGCN, self).__init__()
        self.DEVICE = device
        self.dim_out = dim_out
        self.W_a = nn.Parameter(torch.FloatTensor(self.dim_out, self.dim_out))
        self.U = nn.Parameter(torch.FloatTensor(self.dim_out))
        self.BGCN = BGCN(adj_node, adj_edge, dim_in_node, dim_out, dim_out_edge, M, range_K, device, in_drop=in_drop,
                         gcn_drop=gcn_drop, residual=residual)

    def forward(self, X):
        '''
        X:(B,N,dim_in_node)
        return: h(B,N,dim_out)
        '''
        input = self.BGCN(X)  # (K,B,N,dim_out_node)
        e = torch.einsum('ijkm,m->ijk', torch.einsum('ijkl,lm->ijkm', input, self.W_a),
                         self.U)  # (K,B,N)
        e = e.permute(1, 2, 0)  # (K,B,N)->(B,N,K)
        alpha = F.softmax(e, dim=-1).unsqueeze(-1)
        h = torch.einsum('ijkl,ijlm->ijkm', input.permute(1, 2, 3, 0), alpha).squeeze(-1)
        return h


class MGCN(nn.Module):
    '''
    Multi range Gcn
    计算不同范围GCN
    '''

    def __init__(self, adj_node, adj_edge, dim_in_node, dim_out_node, dim_out_edge, M, range_K, device, in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        '''
        :param range_K: k ranges
        :param adj_node:(V,V)
        :param adj_edge:(E,E)
        :param V:number of node
        :param E:number of edge
        :param M:(V,E),M_{i,(i->j)}=M_{j,(i->j)}=1
        :param dim_in_node: int, num of channels in the input sequence
        :param dim_out_node: int, num of node channels  in the output sequence
        :param dim_out_edge: int, num of edge channels  in the output sequence
        '''
        super(MGCN, self).__init__()
        self.DEVICE = device
        self.K = range_K
        self._M = M
        self.GCN_khops_node = nn.ModuleList(
            [GCN(adj_node, dim_in_node, dim_out_node, k + 1, device, in_drop=in_drop, gcn_drop=gcn_drop,
                 residual=residual) for k in range(self.K)])
        self.linear = nn.Linear(dim_out_node, dim_in_node)

        self.W = nn.Parameter(torch.FloatTensor(dim_in_node, dim_out_node))
        self.b = nn.Parameter(torch.FloatTensor(dim_out_node, ))

    def forward(self, X):
        '''
        计算k个不同范围邻居的GCN
        :param X: (batch_size,N, dim_in_node)
        :return: (K,batch_size,N, dim_out_node)
        '''
        Xs = []
        # Z0 = torch.einsum('ijk,km->ijm', (torch.einsum('ij,jmk->imk', self._M.permute(1, 0), X.permute(1, 0, 2))),
        #                   self.W_b).permute(1, 0, 2)  # (b,E,edge_num_units)
        for k in range(self.K):
            X = self.GCN_khops_node[k](X)
            X = self.linear(X)
            X1 = torch.sigmoid(X.matmul(self.W) + self.b)

            Xs.append(X1)
        Xs = torch.stack(Xs)  # (K,b,V,node_num_units)
        return Xs


class MRA_GCN(nn.Module):
    '''
    计算不同范围邻居的GCN输出的权重
    '''

    def __init__(self, adj_node, adj_edge, dim_in_node, dim_out_node, dim_out_edge, M, range_K, dim_out, device,
                 in_drop=0.0,
                 gcn_drop=0.0, residual=False):
        super(MRA_GCN, self).__init__()
        self.DEVICE = device
        self.dim_out = dim_out
        self.W_a = nn.Parameter(torch.FloatTensor(self.dim_out, self.dim_out))
        self.U = nn.Parameter(torch.FloatTensor(self.dim_out))
        self.MGCN = MGCN(adj_node, adj_edge, dim_in_node, dim_out, dim_out_edge, M, range_K, device, in_drop=in_drop,
                         gcn_drop=gcn_drop, residual=residual)

    def forward(self, X):
        '''
        X:(B,N,dim_in_node)
        return: h(B,N,dim_out)
        '''
        input = self.MGCN(X)  # (K,B,N,dim_out_node)
        e = torch.einsum('ijkm,m->ijk', torch.einsum('ijkl,lm->ijkm', input, self.W_a),
                         self.U)  # (K,B,N)
        e = e.permute(1, 2, 0)  # (K,B,N)->(B,N,K)
        alpha = F.softmax(e, dim=-1).unsqueeze(-1)
        h = torch.einsum('ijkl,ijlm->ijkm', input.permute(1, 2, 3, 0), alpha).squeeze(-1)
        return h