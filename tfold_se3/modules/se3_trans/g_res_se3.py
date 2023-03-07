"""SE(3)-equivariant residual block.

This module computes node features via multi-headed attention mechanism, and adopts the short-cut
  connections to build up a residual block.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from tfold_se3.modules.se3_trans.fiber import Fiber
from tfold_se3.modules.se3_trans.g_conv_se3_partial import GConvSE3Partial
from tfold_se3.modules.se3_trans.g_linear_se3 import GLinearSE3
from tfold_se3.modules.se3_trans.g_mab_se3 import GMABSE3
from tfold_se3.modules.se3_trans.g_sum_se3 import GSumSE3


class GResSE3(nn.Module):
    """SE(3)-equivariant residual block."""

    def __init__(self, f_in, f_out, n_dims_edge, div_fctr=4, n_heads=1):
        """Constructor function.

        Args:
        * f_in: input fiber
        * f_out: output fiber
        * n_dims_edge: number of dimensions for edge embeddings
        * div_fctr: division factor
        * n_heads: number of heads in the multi-headed attention

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.n_dims_edge = n_dims_edge
        self.div_fctr = div_fctr
        self.n_heads = n_heads

        # setup fibers for intermediate features
        struct_dict_mo = {k: max(1, int(v // div_fctr)) for k, v in self.f_out.struct_dict.items()}
        f_mid_out = Fiber(struct_dict=struct_dict_mo)
        struct_dict_mi = {k: v for k, v in struct_dict_mo.items() if k in self.f_in.degrees}
        f_mid_in = Fiber(struct_dict=struct_dict_mi)

        # create modules in the SE(3)-equivariant residual block
        self.conv_qry = GLinearSE3(self.f_in, f_mid_in)  # 1 x 1 convolution
        self.conv_key = GConvSE3Partial(self.f_in, f_mid_in, self.n_dims_edge)
        self.conv_val = GConvSE3Partial(self.f_in, f_mid_out, self.n_dims_edge)
        self.mab = GMABSE3(f_mid_in, f_mid_out, self.n_heads)
        self.proj = GLinearSE3(f_mid_out, self.f_out)  # 1 x 1 convolution
        self.add = GSumSE3(self.f_out, self.f_in)

    def forward(self, graph, feat_dict, basis_dict, radial):
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * feat_dict: dict of node features
        * basis_dict: dict of equivariant bases, indexed by (d_i, d_o)
        * radial: radial distance of each edge in the graph

        Returns:
        * z: dict of output node features
        """

        # calculate query, key, & value embeddings
        q = self.conv_qry(feat_dict)
        k_raw = checkpoint(self.__forward_vk, self.conv_key, graph, feat_dict, basis_dict, radial)
        k = self.__unpack_vk(self.conv_key, k_raw)
        v_raw = checkpoint(self.__forward_vk, self.conv_val, graph, feat_dict, basis_dict, radial)
        v = self.__unpack_vk(self.conv_val, v_raw)

        # aggregate value embeddings via multi-headed attention
        z = self.mab(graph, q, k, v)

        # linear mapping + short-cut connection
        outputs = self.add(self.proj(z), feat_dict)

        return outputs

    @classmethod
    def __forward_vk(cls, model, graph, feat_dict, basis_dict, radial):
        """Perform the forward pass for computing key & value embeddings."""

        v_dict = model(graph, feat_dict, basis_dict, radial)
        v_concat = torch.cat(
            [torch.flatten(v_dict['%d' % d], start_dim=1) for d in model.f_out.degrees], dim=-1)

        return v_concat

    @classmethod
    def __unpack_vk(cls, model, v_concat):
        """Unpack key & value embeddings."""

        split_sizes = [c * (2 * d + 1) for c, d in model.f_out.struct_list]
        v_list = torch.split(v_concat, split_sizes, dim=-1)
        v_dict = {}
        for c, d in model.f_out.struct_list:
            v_dict['%d' % d] = torch.reshape(v_list[d], [-1, c, 2 * d + 1])

        return v_dict

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'GResSE3: ' + ', '.join([
            'f_in={%s}' % str(self.f_in),
            'f_out={%s}' % str(self.f_out),
            'n_dims_edge=%d' % self.n_dims_edge,
            'div_fctr=%d' % self.div_fctr,
            'n_heads=%d' % self.n_heads,
        ])

        return repr_str
