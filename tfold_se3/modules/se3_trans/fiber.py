"""Data structure for defining the dimension of multi-degree features."""

import torch


class Fiber():
    """Data structure for defining the dimension of multi-degree features.

    Note: this class only define the feature dimension, rather than storing actual features.
    """

    def __init__(self,
                 n_degrees=None,
                 n_channels=None,
                 struct_list=None,
                 struct_dict=None,
        ):
        """Constructor function.

        Args:
        * n_degrees: number of degrees (0, 1, ..., n_degrees - 1)
        * n_channels: number of channels per degree
        * struct_list: list of (n_channels, degree) pairs
        * struct_dict: dict of number of channels, indexed by the degree

        Returns: n/a
        """

        # setup feature dimensions
        if struct_list is not None:
            self.struct_list = struct_list
        elif struct_dict is not None:
            self.struct_list = [(struct_dict[k], k) for k in sorted(struct_dict.keys())]
        else:
            self.struct_list = [(n_channels, degree) for degree in range(n_degrees)]

        # setup additional attributes
        _, self.degrees = zip(*self.struct_list)  # list of degree indices
        self.degree_min = min(self.degrees)
        self.degree_max = max(self.degrees)
        self.struct_dict = {k: v for v, k in self.struct_list}

        # determine feature dimension indices for each degree
        self.idxs_dim_dict = {}
        for n_chns, degree in self.struct_list:
            idx_dim_base = 0 if degree == 0 else self.idxs_dim_dict[degree - 1][1]
            n_dims_curr = n_chns * (2 * degree + 1)
            self.idxs_dim_dict[degree] = (idx_dim_base, idx_dim_base + n_dims_curr)
        self.n_dims = self.idxs_dim_dict[self.degree_max][1]  # maximal degree's upper bound

    @staticmethod
    def combine_max(fiber_1, fiber_2):
        """Fuse two fibers by taking the maximal number of channels for each degree.

        Args:
        * fiber_1: the first fiber
        * fiber_2: the second fiber

        Returns:
        * fiber_out: the output fiber
        """

        struct_dict = {}
        degrees = set(fiber_1.degrees) | set(fiber_2.degrees)
        for degree in degrees:
            if degree not in fiber_1.struct_dict:
                struct_dict[degree] = fiber_2.struct_dict[degree]
            elif degree not in fiber_2.struct_dict:
                struct_dict[degree] = fiber_1.struct_dict[degree]
            else:
                struct_dict[degree] = max(fiber_1.struct_dict[degree], fiber_2.struct_dict[degree])
        fiber_out = Fiber(struct_dict=struct_dict)

        return fiber_out

    def __repr__(self):
        """Get the string representation."""

        return ', '.join(['%d: %d' % (k, self.struct_dict[k]) for k in self.degrees])


def fiber2head(feat_dict, n_heads, fiber):
    """Convert multi-degree features into a multi-head tensor.

    Args:
    * feat_dict: dict of multi-degree features, indexed by the degree
    * n_heads: number of heads in the output tensor
    * fiber: multi-degree features's dimension configuration

    Returns:
    * feat_tns: multi-head tensor
    """

    feat_list = []
    for d in fiber.degrees:
        feat_list.append(feat_dict['%d' % d].view(*feat_dict['%d' % d].shape[:-2], n_heads, -1))
    feat_tns = torch.cat(feat_list, dim=-1)

    return feat_tns
