"""SE(3)-equivariant linear layer (equivalent to 1x1 convolution).

This module performs linear mapping for each feature type. The input features can either be node or
  edge features.
"""

import numpy as np
import torch
from torch import nn


class GLinearSE3(nn.Module):
    """SE(3)-equivariant linear layer (equivalent to 1x1 convolution)."""

    def __init__(self, f_in, f_out):
        """Constructor function.

        Args:
        * f_in: input fiber
        * f_out: output fiber

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out

        # create parameters for linear mappgings, one per output feature degree
        self.param_dict = nn.ParameterDict()
        for co, do in self.f_out.struct_list:
            ci = self.f_in.struct_dict[do]
            self.param_dict['%d' % do] = nn.Parameter(torch.randn(co, ci) / np.sqrt(ci))

    def forward(self, feat_dict):
        """Perform the forward pass.

        Args:
        * feat_dict: dict of node/edge features

        Returns:
        * outputs: dict of output node/edge features
        """

        outputs = {}
        for key, val in feat_dict.items():
            outputs[key] = torch.matmul(self.param_dict[key], val)

        return outputs

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'GLinearSE3: ' + ', '.join([
            'f_in={%s}' % str(self.f_in),
            'f_out={%s}' % str(self.f_out),
        ])

        return repr_str
