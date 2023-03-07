"""SE(3)-equivariant ReLU layer.

This module applies ReLU non-linearity to each component in the feature dict.
"""

import numpy as np
import torch
from torch import nn


class GReLUSE3(nn.Module):
    """SE(3)-equivariant ReLU layer."""

    def __init__(self):
        """Constructor function."""

        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, feat_dict):
        """Perform the forward pass.

        Args:
        * feat_dict: dict of node/edge features

        Returns:
        * outputs: dict of output node/edge features
        """

        outputs = {}
        for key, val in feat_dict.items():
            outputs[key] = self.relu(val)

        return outputs

    def __repr__(self):
        """Get the string representation."""

        return 'GReLUSE3'


def unit_test():
    """Unit-tests for the <GReLUSE3> module."""

    # configurations
    batch_size = 16
    n_degrees = 3
    n_channels = 8

    # test <GBatchNormSE3>
    inputs = {}
    for degree in range(n_degrees):
        feat_tns_np = np.random.normal(size=(batch_size, n_channels, 2 * degree + 1))
        inputs['%d' % degree] = torch.tensor(feat_tns_np, dtype=torch.float32)
    layer = GReLUSE3()
    outputs = layer(inputs)
    for key, val in outputs.items():
        print('{} => {}'.format(key, val.shape))


if __name__ == '__main__':
    unit_test()
