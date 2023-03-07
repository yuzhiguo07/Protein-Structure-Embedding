"""SE(3)-equivariant conditional batch normalization layer.

This module applies conditional batch normalization to input features.
Unlike <GNormSE3>, ReLU non-linearity is not included in this module.
"""

import torch
from torch import nn


class GCondBatchNormSE3(nn.Module):
    """SE(3)-equivariant conditional batch normalization layer."""

    def __init__(self, fiber, cond_depth, bias=True):
        """Constructor function.

        Args:
        * fiber: input/output fiber
        * cond_depth: number of (gamma, beta) pairs
        * bias: whether to include the bias term

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.fiber = fiber
        self.cond_depth = cond_depth
        self.bias = bias
        self.eps = 1e-12

        # build normalization & non-linear activations layers, one per output feature degree
        self.norm_dict = nn.ModuleDict()
        for n_chns, degree in self.fiber.struct_list:
            self.norm_dict['%d' % degree] = nn.BatchNorm1d(n_chns, affine=False)

        # create multiple (gamma, beta) pairs
        self.embed_dict = nn.ModuleDict()
        for n_chns, degree in self.fiber.struct_list:
            self.embed_dict['g-%d' % degree] = nn.Embedding(cond_depth, n_chns)
            self.embed_dict['g-%d' % degree].weight.data.uniform_()
            if self.bias:
                self.embed_dict['b-%d' % degree] = nn.Embedding(cond_depth, n_chns)
                self.embed_dict['b-%d' % degree].weight.data.zero_()

    def forward(self, feat_dict, cond_idxs):
        """Perform the forward pass.

        Args:
        * feat_dict: dict of node features
        * cond_idxs: conditional indices of size N_v

        Returns:
        * outputs: dict of output node features
        """

        outputs = {}
        for key, val in feat_dict.items():
            n_chns = val.shape[1]
            gamma = self.embed_dict['g-' + key](cond_idxs)
            norm = torch.norm(val, dim=-1, keepdim=True).clamp_min(self.eps)
            scale = self.norm_dict[key](norm[..., 0]).unsqueeze(-1)
            outputs[key] = gamma.view(-1, n_chns, 1) * scale * (val / norm)
            if self.bias:
                beta = self.embed_dict['b-' + key](cond_idxs)
                outputs[key] += beta.view(-1, n_chns, 1)

        return outputs

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'GCondBatchNormSE3: ' + ', '.join([
            'fiber={%s}' % str(self.fiber),
            'cond_depth=%d' % self.cond_depth,
            'bias=%s' % str(self.bias),
        ])

        return repr_str
