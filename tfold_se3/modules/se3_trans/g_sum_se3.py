"""SE(3)-equivariant summation layer.

This module sum up input features of each input degree, with zero-padding applied when necessary.
  The input features can either be node or edge features.
"""

from torch import nn
import torch.nn.functional as F

from tfold_se3.modules.se3_trans.fiber import Fiber


class GSumSE3(nn.Module):
    """SE(3)-equivariant summation layer."""

    def __init__(self, f_in1, f_in2):
        """Constructor function.

        Args:
        * f_in1: fiber of the 1st dict of input features to be summed
        * f_in2: fiber of the 2nd dict of input features to be summed

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.f_in1 = f_in1
        self.f_in2 = f_in2
        self.f_out = Fiber.combine_max(f_in1, f_in2)

    def forward(self, in1, in2):
        """Perform the forward pass.

        Args:
        * in1: 1st dict of input features
        * in2: 2nd dict of input features

        Returns:
        * out: dict of output features
        """

        out = {}
        for d in self.f_out.degrees:
            key = '%d' % d
            if not (key in in1 and key in in2):
                if key in in1:
                    out[key] = in1[key]
                else:  # then <key> must be in <in2>
                    out[key] = in2[key]
            else:
                n_chns_out = max(in1[key].shape[1], in2[key].shape[1])
                in1_pad = in1[key] if in1[key].shape[1] == n_chns_out else \
                    F.pad(in1[key], (0, 0, 0, n_chns_out - in1[key].shape[1]))
                in2_pad = in2[key] if in2[key].shape[1] == n_chns_out else \
                    F.pad(in2[key], (0, 0, 0, n_chns_out - in2[key].shape[1]))
                out[key] = in1_pad + in2_pad

        return out

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'GSum: ' + ', '.join([
            'f_in1={%s}' % str(self.f_in1),
            'f_in2={%s}' % str(self.f_in2),
        ])

        return repr_str
