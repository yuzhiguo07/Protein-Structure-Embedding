"""SE(3)-equivariant normalization layer.

This module normalizes input features, followed by a ReLU activation layer.

Note:
* In SE(3)-Transformer's original implementation, a MLP network (BN + ReLU + Linear) can be inserted
    before the final normalization & non-linear activation layers. However, the default behavior is
    not to use this MLP network. Here, we simplify the implementation to disable this feature.
"""

import torch
from torch import nn


class GNormSE3(nn.Module):
    """SE(3)-equivariant normalization layer."""

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
            self.norm_dict['%d' % degree] = nn.Sequential(
                nn.LayerNorm(n_chns),
                nn.ReLU(),
            )

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

        repr_str = 'GNormSE3: ' + ', '.join([
            'fiber={%s}' % str(self.fiber),
        ])

        return repr_str
