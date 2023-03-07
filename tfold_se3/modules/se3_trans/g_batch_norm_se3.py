"""SE(3)-equivariant batch normalization layer.

This module applies batch normalization to input features.
Unlike <GNormSE3>, ReLU non-linearity is not included in this module.
"""

import torch
from torch import nn


class GBatchNormSE3(nn.Module):
    """SE(3)-equivariant batch normalization layer."""

    def __init__(self, fiber):
        """Constructor function.

        Args:
        * fiber: input/output fiber

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.fiber = fiber
        self.eps = 1e-12

        # build normalization & non-linear activations layers, one per output feature degree
        self.norm_dict = nn.ModuleDict()
        for n_chns, degree in self.fiber.struct_list:
            self.norm_dict['%d' % degree] = nn.BatchNorm1d(n_chns)

    def forward(self, feat_dict):
        """Perform the forward pass.

        Args:
        * feat_dict: dict of node/edge features

        Returns:
        * outputs: dict of output node/edge features
        """

        outputs = {}
        for key, val in feat_dict.items():
            norm = torch.norm(val, dim=-1, keepdim=True).clamp_min(self.eps)
            scale = self.norm_dict[key](norm[..., 0]).unsqueeze(-1)
            outputs[key] = scale * (val / norm)

        return outputs

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'GBatchNormSE3: ' + ', '.join([
            'fiber={%s}' % str(self.fiber),
        ])

        return repr_str
