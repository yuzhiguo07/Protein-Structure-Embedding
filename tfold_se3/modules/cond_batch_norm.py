"""Conditional batch normalization layers for 1D/2D features.

In a vanilla batch normalization layer, only one (gamma, beta) pair is used to apply linear mapping
  on normalized 1D/2D features.
In a conditional batch normalization layer, multiple (gamma, beta) pairs are used to apply different
  linear mappings under different 'conditions'. The 'condition' is represented by an integer,
  ranging from 0 to K - 1, and fed into the conditional batch normalization layer, along with 1D/2D
  features.
"""

import numpy as np
import torch
from torch import nn


class CondBatchNorm1d(nn.Module):
    """Conditional batch normalization layer for 1D features."""

    def __init__(self, n_dims, depth, bias=True):
        """Constructor function.

        Args:
        * n_dims: number of feature dimensions
        * depth: number of (gamma, beta) pairs
        * bias: whether to include the bias term

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.n_dims = n_dims
        self.bias = bias

        # create a vanilla batch normalization layer and multiple (gamma, beta) pairs
        self.norm = nn.BatchNorm1d(n_dims, affine=False)
        self.embed_gamma = nn.Embedding(depth, n_dims)
        self.embed_gamma.weight.data.normal_(1.0, 0.01)
        if self.bias:
            self.embed_beta = nn.Embedding(depth, n_dims)
            self.embed_beta.weight.data.zero_()

    def forward(self, inputs, idxs):
        """Perform the forward pass.

        Args:
        * inputs: input feature maps of size BS x D x L
        * idxs: conditional indices of size BS

        Returns:
        * outputs: output feature maps of size BS x D x L
        """

        outputs = self.embed_gamma(idxs).view(-1, self.n_dims, 1) * self.norm(inputs)
        if self.bias:
            outputs += self.embed_beta(idxs).view(-1, self.n_dims, 1)

        return outputs


class CondBatchNorm2d(nn.Module):
    """Conditional batch normalization layer for 2D features."""

    def __init__(self, n_chns, depth, bias=True):
        """Constructor function.

        Args:
        * n_chns: number of feature map channels
        * depth: number of (gamma, beta) pairs
        * bias: whether to include the bias term

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.n_chns = n_chns
        self.bias = bias

        # create a vanilla batch normalization layer and multiple (gamma, beta) pairs
        self.norm = nn.BatchNorm2d(n_chns, affine=False)
        self.embed_gamma = nn.Embedding(depth, n_chns)
        self.embed_gamma.weight.data.normal_(1.0, 0.01)
        if self.bias:
            self.embed_beta = nn.Embedding(depth, n_chns)
            self.embed_beta.weight.data.zero_()

    def forward(self, inputs, idxs):
        """Perform the forward pass.

        Args:
        * inputs: input feature maps of size BS x C x H x W
        * idxs: conditional indices of size BS

        Returns:
        * outputs: output feature maps of size BS x C x H x W
        """

        outputs = self.embed_gamma(idxs).view(-1, self.n_chns, 1, 1) * self.norm(inputs)
        if self.bias:
            outputs += self.embed_beta(idxs).view(-1, self.n_chns, 1, 1)

        return outputs


class CondBatchNorm2dLegacy(nn.Module):
    """Conditional batch normalization layer - the legacy version."""

    def __init__(self, n_chns, depth, bias=True):
        """Constructor function."""

        super().__init__()

        self.n_chns = n_chns
        self.bias = bias
        self.bn = nn.BatchNorm2d(n_chns, affine=False)
        if self.bias:
            self.embed = nn.Embedding(depth, n_chns * 2)
            self.embed.weight.data[:, :n_chns].uniform_()
            self.embed.weight.data[:, n_chns:].zero_()
        else:
            self.embed = nn.Embedding(depth, n_chns)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        """Perform the forward pass."""

        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.n_chns, 1, 1) * out + beta.view(-1, self.n_chns, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.n_chns, 1, 1) * out

        return out


def unit_test():
    """Run unit-tests for <CondBatchNorm1d> & <CondBatchNorm2d>."""

    # configurations
    batch_size = 4
    n_chns = 32  # number of channels of 2D features
    seq_len = 64  # sequence length of 1D features
    height = 16  # spatial height of 2D features
    width = 16  # spatial width of 2D features
    depth = 8

    # test <CondBatchNorm1d>
    inputs_np = np.random.normal(size=(batch_size, n_chns, seq_len)).astype(np.float32)
    idxs_np = np.random.randint(depth, size=(batch_size)).astype(np.int64)
    inputs = torch.tensor(inputs_np, dtype=torch.float32)
    idxs = torch.tensor(idxs_np, dtype=torch.int64)  # indices must be int64
    module = CondBatchNorm1d(n_chns, depth, bias=True)
    outputs = module(inputs, idxs)
    print('outputs: {}'.format(outputs.shape))

    # test <CondBatchNorm2d>
    inputs_np = np.random.normal(size=(batch_size, n_chns, height, width)).astype(np.float32)
    idxs_np = np.random.randint(depth, size=(batch_size)).astype(np.int64)
    inputs = torch.tensor(inputs_np, dtype=torch.float32)
    idxs = torch.tensor(idxs_np, dtype=torch.int64)  # indices must be int64
    module = CondBatchNorm2d(n_chns, depth, bias=True)
    outputs = module(inputs, idxs)
    print('outputs: {}'.format(outputs.shape))

    # test <CondBatchNorm2dLegacy>
    inputs_np = np.random.normal(size=(batch_size, n_chns, height, width)).astype(np.float32)
    idxs_np = np.random.randint(depth, size=(batch_size)).astype(np.int64)
    inputs = torch.tensor(inputs_np, dtype=torch.float32)
    idxs = torch.tensor(idxs_np, dtype=torch.int64)  # indices must be int64
    module = CondBatchNorm2dLegacy(n_chns, depth, bias=True)
    outputs = module(inputs, idxs)
    print('outputs: {}'.format(outputs.shape))


if __name__ == '__main__':
    unit_test()
