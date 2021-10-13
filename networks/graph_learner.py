import torch
from torch import nn, Tensor
from torch.nn import functional as F
#import heatmap

class MiLearner(nn.Module):
    def __init__(self, n_hist: int, n_in: int, node_dim: int, dropout: float):
        super(MiLearner, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.nodes = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(1, 1)), 
            nn.ReLU(True),
            nn.Conv2d(64, node_dim, kernel_size=(1, n_hist))
        )
      

    def forward(self, inputs: Tensor):
        """
        :param inputs: tensor, [B, T, N, F]
        :param supports: tensor, [E, N, N]
        :return: tensor, [E, B, N, N]
        """
        x = inputs.transpose(1, 3)  
        nodes = self.nodes(x).squeeze(3).transpose(1, 2)  
       
        self.dropout(nodes)
        
        m = nodes
        A_mi = torch.einsum('bud,bvd->buv', [m, m])
        return  A_mi


class GraphLearner(nn.Module):
    def __init__(self, supports: Tensor, n_hist: int, n_in: int, node_dim: int, dropout: float, learn_macro: bool, learn_micro: bool):
        super(GraphLearner, self).__init__()
        self.adaptive = nn.Parameter(supports, requires_grad=learn_macro)

        if learn_micro:
            self.mi_learner = MiLearner(n_hist, n_in, node_dim, dropout)

    def forward(self, inputs: Tensor = None) -> Tensor:
        """
        :param inputs: tensor, [B, T, N, F]
        :return: tensor, [E, N, N] or [E, B, N, N]
        """
        supports = self.adaptive

        if hasattr(self, 'mi_learner'):
            #Multi-level Graph Structure Fusion
            supports = supports.unsqueeze(1) + self.mi_learner(inputs)
    
        return  F.normalize(torch.relu(supports), p=1, dim=-1)


