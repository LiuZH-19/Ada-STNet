import torch
from torch import nn, Tensor
from torch.nn import functional as F

from utils.graph import GraphConv


class STLayer(nn.Module):
    def __init__(self, n_residuals: int, n_dilations: int, kernel_size: int, dilation: int,
                 n_skip: int, edge_dim: int, dropout: float):
        super(STLayer, self).__init__()
        # dilated convolutions
        self.filter_conv = nn.Conv2d(n_residuals, n_dilations, kernel_size=(1, kernel_size), dilation=dilation)
        self.gate_conv = nn.Conv1d(n_residuals, n_dilations, kernel_size=(1, kernel_size), dilation=dilation)

        # 1x1 convolution for residual connection
        self.gconv = GraphConv(n_dilations, n_residuals, edge_dim)
        self.dropout = nn.Dropout(dropout, inplace=True)

        # 1x1 convolution for skip connection
        self.skip_conv = nn.Conv1d(n_dilations, n_skip, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(n_residuals)

    def forward(self, x: Tensor, skip: Tensor, supports: Tensor):
        residual = x # [B, F, N, T]
        # dilated convolution
        _filter = self.filter_conv(residual)
        _filter = torch.tanh(_filter)
        _gate = self.gate_conv(residual)
        _gate = torch.sigmoid(_gate)
        x = _filter * _gate  

        # parametrized skip connection
        s = x
        s = self.skip_conv(s)
        skip = skip[:, :, :, -s.size(3):]
        skip = s + skip

        x = self.gconv(x, supports) 
        self.dropout(x)

        x = x + residual[:, :, :, -x.size(3):]

        x = self.bn(x)
        return x, skip


class STBlock(nn.ModuleList):
    def __init__(self, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int,
                 n_skips: int, edge_dim: int, dropout: float):
        super(STBlock, self).__init__()
        for i in range(n_layers):
            self.append(
                STLayer(n_residuals, n_dilations, kernel_size, 2 ** i, n_skips, edge_dim, dropout)
            )

    def forward(self, x: Tensor, skip: Tensor, supports: Tensor):
        for layer in self:
            x, skip = layer(x, skip, supports)

        return x, skip


class StackedSTBlocks(nn.ModuleList):
    def __init__(self, n_blocks, n_layers: int, kernel_size: int, n_residuals: int, n_dilations: int,
                 n_skips: int, edge_dim: int, dropout: float):
        self.n_skips = n_skips
        super(StackedSTBlocks, self).__init__()
        for _ in range(n_blocks):
            self.append(
                STBlock(n_layers, kernel_size, n_residuals, n_dilations, n_skips, edge_dim, dropout))

    def forward(self, x: Tensor, supports: Tensor):
        b, f, n, t = x.shape
        skip = torch.zeros(b, self.n_skips, n, t, dtype=torch.float32, device=x.device)
        for block in self:
            x, skip = block(x, skip, supports)
        return x, skip


class Predictor(nn.Module):
    def __init__(self,
                 n_in: int,
                 n_out: int,
                 n_pred: int,
                 edge_dim: int,
                 n_residuals: int,
                 n_dilations: int,
                 n_skips: int,
                 n_ends: int,
                 kernel_size: int,
                 n_blocks: int,
                 n_layers: int,
                 dropout: float):
        super(Predictor, self).__init__()
        # n_in = n_in + 2
        self.t_pred = n_pred
        
        # the reduction in the time dimension after stackedSTBlocks 
        self.receptive_field = n_blocks * (kernel_size - 1) * (2 ** n_layers - 1) + 1
       
        #fully connected networks for expanding the feature dimension
        self.enter = nn.Conv2d(n_in, n_residuals, kernel_size=(1, 1))

        self.blocks = StackedSTBlocks(n_blocks, n_layers, kernel_size, n_residuals, n_dilations,
                                      n_skips, edge_dim, dropout)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_skips, n_ends, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(n_ends, n_pred * n_out, kernel_size=(1, 1))
        )

    def forward(self, inputs: Tensor, supports: Tensor):
        """
        : params inputs: tensor, [B, T, N, F]
        """
        inputs = inputs.transpose(1, 3) 

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            x = F.pad(inputs, [self.receptive_field - in_len, 0, 0, 0])
        else:
            x = inputs
        x = self.enter(x)

        b, c, n, t = x.shape

        x, skip = self.blocks(x, supports)

        y_ = self.out(skip)
        return y_.reshape(b, self.t_pred, -1, n).transpose(-1, -2)


