"""Conditional instance normalization layers for 1D/2D features.

In a vanilla instance normalization layer, only one (gamma, beta) pair is used to apply linear
  mapping on normalized 1D/2D features.
In a conditional instance normalization layer, multiple (gamma, beta) pairs are used to apply
  different linear mappings under different 'conditions'. The 'condition' is represented by an
  integer, ranging from 0 to K - 1, and fed into the conditional instance normalization layer, along
  with 1D/2D features.

We further allow the usage of additional term (as in CondInstanceNorm++) to account for the color-
  shifting issue in the original CondInstanceNorm module. This can be enabled by setting the
  <fix_color_shift> argument to True (default: False).
"""

import numpy as np
import torch
from torch import nn


class CondInstanceNorm1d(nn.Module):
    """Conditional instance normalization layer for 1D features."""

    def __init__(self, n_dims, depth, bias=True, fix_color_shift=False, track_running_stats=False):
        """Constructor function.

        Args:
        * n_dims: number of feature dimensions
        * depth: number of (gamma, beta) pairs
        * bias: whether to include the bias term
        * fix_color_shift: whether to fix the color shifting issue
        * track_running_stats: whether to track the running mean and variance for normalization

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.n_dims = n_dims
        self.bias = bias
        self.fix_color_shift = fix_color_shift

        # create a vanilla batch normalization layer and multiple (gamma, beta) pairs
        self.norm = nn.InstanceNorm1d(n_dims, affine=False, track_running_stats=track_running_stats)
        self.embed_gamma = nn.Embedding(depth, n_dims)
        self.embed_gamma.weight.data.normal_(1.0, 0.01)
        if self.bias:
            self.embed_beta = nn.Embedding(depth, n_dims)
            self.embed_beta.weight.data.zero_()
        if self.fix_color_shift:
            self.embed_alpha = nn.Embedding(depth, n_dims)
            self.embed_alpha.weight.data.normal_(1.0, 0.01)

    def forward(self, inputs, idxs):
        """Perform the forward pass.

        Args:
        * inputs: input feature maps of size BS x D x L
        * idxs: conditional indices of size BS

        Returns:
        * outputs: output feature maps of size BS x D x L
        """

        if self.fix_color_shift:
            mean_mat_raw = torch.mean(inputs, dim=2)  # BS x D
            m = torch.mean(mean_mat_raw, dim=-1, keepdim=True)  # BS
            v = torch.std(mean_mat_raw, dim=-1, keepdim=True)  # BS
            mean_mat_nrm = (mean_mat_raw - m) / (v + 1e-4)  # BS x D

        outputs = self.embed_gamma(idxs).view(-1, self.n_dims, 1) * self.norm(inputs)
        if self.bias:
            outputs += self.embed_beta(idxs).view(-1, self.n_dims, 1)
        if self.fix_color_shift:
            outputs += (self.embed_alpha(idxs) * mean_mat_nrm).view(-1, self.n_dims, 1)

        return outputs


class CondInstanceNorm2d(nn.Module):
    """Conditional instance normalization layer for 2D features."""

    def __init__(self, n_chns, depth, bias=True, fix_color_shift=False, track_running_stats=False):
        """Constructor function.

        Args:
        * n_chns: number of feature map channels
        * depth: number of (gamma, beta) pairs
        * bias: whether to include the bias term
        * fix_color_shift: whether to fix the color shifting issue
        * track_running_stats: whether to track the running mean and variance for normalization

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.n_chns = n_chns
        self.bias = bias
        self.fix_color_shift = fix_color_shift

        # create a vanilla batch normalization layer and multiple (gamma, beta) pairs
        self.norm = nn.InstanceNorm2d(n_chns, affine=False, track_running_stats=track_running_stats)
        self.embed_gamma = nn.Embedding(depth, n_chns)
        self.embed_gamma.weight.data.normal_(1.0, 0.01)
        if self.bias:
            self.embed_beta = nn.Embedding(depth, n_chns)
            self.embed_beta.weight.data.zero_()
        if self.fix_color_shift:
            self.embed_alpha = nn.Embedding(depth, n_chns)
            self.embed_alpha.weight.data.normal_(1.0, 0.01)

    def forward(self, inputs, idxs):
        """Perform the forward pass.

        Args:
        * inputs: input feature maps of size BS x C x H x W
        * idxs: conditional indices of size BS

        Returns:
        * outputs: output feature maps of size BS x C x H x W
        """

        if self.fix_color_shift:
            mean_mat_raw = torch.mean(inputs, dim=(2, 3))  # BS x C
            m = torch.mean(mean_mat_raw, dim=-1, keepdim=True)  # BS
            v = torch.std(mean_mat_raw, dim=-1, keepdim=True)  # BS
            mean_mat_nrm = (mean_mat_raw - m) / (v + 1e-4)  # BS x C

        outputs = self.embed_gamma(idxs).view(-1, self.n_chns, 1, 1) * self.norm(inputs)
        if self.bias:
            outputs += self.embed_beta(idxs).view(-1, self.n_chns, 1, 1)
        if self.fix_color_shift:
            outputs += (self.embed_alpha(idxs) * mean_mat_nrm).view(-1, self.n_chns, 1, 1)

        return outputs


def unit_test():
    """Run unit-tests for <CondInstanceNorm1d> & <CondInstanceNorm2d>."""

    # configurations
    batch_size = 4
    n_chns = 32  # number of channels of 2D features
    seq_len = 64  # sequence length of 1D features
    height = 16  # spatial height of 2D features
    width = 16  # spatial width of 2D features
    depth = 8

    # test <CondInstanceNorm1d>
    inputs_np = np.random.normal(size=(batch_size, n_chns, seq_len)).astype(np.float32)
    idxs_np = np.random.randint(depth, size=(batch_size)).astype(np.int64)
    inputs = torch.tensor(inputs_np, dtype=torch.float32)
    idxs = torch.tensor(idxs_np, dtype=torch.int64)  # indices must be int64
    module = CondInstanceNorm1d(n_chns, depth, bias=True, fix_color_shift=True)
    outputs = module(inputs, idxs)
    print('outputs: {}'.format(outputs.shape))

    # test <CondInstanceNorm2d>
    inputs_np = np.random.normal(size=(batch_size, n_chns, height, width)).astype(np.float32)
    idxs_np = np.random.randint(depth, size=(batch_size)).astype(np.int64)
    inputs = torch.tensor(inputs_np, dtype=torch.float32)
    idxs = torch.tensor(idxs_np, dtype=torch.int64)  # indices must be int64
    module = CondInstanceNorm2d(n_chns, depth, bias=True, fix_color_shift=True)
    outputs = module(inputs, idxs)
    print('outputs: {}'.format(outputs.shape))


if __name__ == '__main__':
    unit_test()
