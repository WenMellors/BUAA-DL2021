import numpy as np
def get_adj_edge(adj_mx):
    n = adj_mx.shape[0]
    m = 0
    dict = {}
    for i in range(n):
        for j in range(i + 1,n):
            if adj_mx[i][j] != 0:
                dict[i * i * i + j * j * j] = m
                m = m + 1
    W = np.zeros((m, m))
    for i in range(n):
        for j in range(i + 1,n):
            for k in range(j + 1,n):
                if adj_mx[i][j] != 0 and adj_mx[j][k] != 0:
                    W[dict.get(i*i*i+j*j*j)][dict.get(j*j*j+k*k*k)] = 1
    # print(W)
    return W


def get_M(adj_mx):
    n = adj_mx.shape[0]
    # print(n)
    m = 0
    for i in range(n):
        for j in range(i + 1,n):
            if adj_mx[i][j] != 0:
                m = m + 1
    # print(m)
    k = 0
    W = np.zeros((n,m))
    for i in range(n):
        for j in range(i + 1,n):
            if adj_mx[i][j] != 0:
                W[i][k] = 1
                W[j][k] = 1
                k = k + 1
    # print(W)
    return W

