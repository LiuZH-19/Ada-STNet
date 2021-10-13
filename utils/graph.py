import os
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import eigsh
from torch import sparse, nn, Tensor

from .utils import load_pickle


def normalized_laplacian(w: np.ndarray) -> sp.coo_matrix:
    w = sp.coo_matrix(w)
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(w.shape[0]) - w.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt).tocoo()


def random_walk_matrix(w) -> sp.coo_matrix:
    w = sp.coo_matrix(w)
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(w).tocoo()


def scaled_laplacian(w: np.ndarray, lambda_max: Optional[float] = 2., undirected: bool = True) -> sp.coo_matrix:
    if undirected:
        w = np.maximum.reduce([w, w.T])
    lp = normalized_laplacian(w)
    if lambda_max is None:
        lambda_max, _ = eigsh(lp.todense(), 1, which='LM')
        lambda_max = lambda_max[0]
    lp = sp.csr_matrix(lp)
    m, _ = lp.shape
    i = sp.identity(m, format='csr', dtype=lp.dtype)
    lp = (2 / lambda_max * lp) - i
    return lp.astype(np.float32).tocoo()


def cheb_poly_approx(lp, k_hop, n):
    """
    Chebyshev polynomials approximation function.
    :param lp: np.ndarray, [n_route, n_route], graph Laplacian.
    :param k_hop: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks, n_route].
    """
    l0, l1 = np.identity(n), np.copy(lp)

    if k_hop > 1:
        l_list = [np.copy(l0), np.copy(l1)]
        for i in range(k_hop - 2):
            ln = 2 * np.matmul(lp, l1) - l0
            l_list.append(np.copy(ln))
            l0, l1 = np.copy(l1), np.copy(ln)
        # L_lsit Ks, [n, n], [n, Ks, n]
        return np.stack(l_list, axis=1)
    elif k_hop == 1:
        return l0.reshape((n, 1, n))
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{k_hop}".')


def first_approx(w, n):
    """
    1st-order approximation function.
    :param w: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    """
    a = w + np.identity(n)
    d = np.sum(a, axis=1)
    sinv_d = np.sqrt(np.linalg.inv(np.diag(d)))
    # refer to Eq.5
    return np.identity(n) + np.matmul(np.matmul(sinv_d, a), sinv_d)


def sparse_scipy2torch(w: sp.coo_matrix):
    """
    build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    :return:
    """
    shape = w.shape
    i = torch.tensor(np.vstack((w.row, w.col)).astype(int)).long()
    v = torch.tensor(w.data).float()
    return sparse.FloatTensor(i, v, torch.Size(shape))


def load_graph_data(dataset: str, graph_type: str) -> List[sp.coo_matrix]:
    _, _, adj_mx = load_pickle(os.path.join('data', dataset, 'adj_mx.pkl'))
    if graph_type == 'raw':
        adj = [sp.coo_matrix(adj_mx)]
    elif graph_type == "scalap":
        adj = [scaled_laplacian(adj_mx)]
    elif graph_type == "normlap":
        adj = [normalized_laplacian(adj_mx)]
    elif graph_type == "transition":
        adj = [random_walk_matrix(adj_mx)]
    elif graph_type == "doubletransition":
        adj = [random_walk_matrix(adj_mx), random_walk_matrix(adj_mx.T)]
    elif graph_type == "identity":
        adj = [sp.identity(adj_mx.shape[0], dtype=np.float32, format='coo')]
    else:
        raise ValueError(f"graph type {graph_type} not defined")
    return adj


class GraphConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, edge_dim: int):
        super(GraphConv, self).__init__()
        self.c_in, self.c_out, self.edge_dim = c_in, c_out, edge_dim
        self.out = nn.Conv2d(c_in * (edge_dim + 1), c_out, kernel_size=(1, 1))

    def forward(self, x: Tensor, supports: Tensor):
        """
        :param x: tensor, [B, c_in, N, T] or [B, c_in, N]
        :param supports: tensor, [n_edge, N, N] or [n_edge, B, N, N]
        :return: tensor, [B, c_out, N, T] or [B, c_out, N]
        """
        flag = (len(x.shape) == 3)
        if flag:
            x.unsqueeze_(-1)

        h = [x] + [self.nconv(x, a) for a in supports]   #formula (21) in the paper 
        h = torch.cat(h, 1) #[B, cin*(edge_dim+1), N, T] or [B, c_in*(edge_dim+1), N]
        return self.out(h).squeeze(-1) if flag else self.out(h)

    @staticmethod
    def nconv(x: Tensor, a: Tensor):
        """
        :param x: tensor, [B, C, N, T]
        :param a: tensor, [B, N, N] or [N, N]
        :return:
        """
        a_ = 'vw' if len(a.shape) == 2 else 'bvw'
        x = torch.einsum(f'bcvt,{a_}->bcwt', [x, a])  # x_bcwt = sum_v(x_bcvt * a_bvw)
        return x.contiguous()

    def __repr__(self):
        return f'GraphConv({self.c_in}, {self.c_out}, edge_dim={self.edge_dim}, attn={hasattr(self, "attn")})'
