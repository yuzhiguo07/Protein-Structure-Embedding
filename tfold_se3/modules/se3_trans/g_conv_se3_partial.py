"""SE(3)-equivariant partial convolutional layer.

This module computes a dict of edge features by summing over <PairwiseConv>'s outputs of all the
  input feature types.
"""

import torch
from torch import nn

from tfold_se3.modules.se3_trans.pairwise_conv import PairwiseConv


class GConvSE3Partial(nn.Module):
    """SE(3)-equivariant partial convolutional layer."""

    def __init__(self, f_in, f_out, n_dims_edge=0):
        """Constructor function.

        Args:
        * f_in: input fiber
        * f_out: output fiber
        * n_dims_edge: number of dimensions for edge embeddings

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.n_dims_edge = n_dims_edge

        # create a PairwiseConv operation for each (d_i, d_o) pair
        self.pconv_dict = nn.ModuleDict()
        for ci, di in self.f_in.struct_list:
            for co, do in self.f_out.struct_list:
                self.pconv_dict['%d-%d' % (di, do)] = PairwiseConv(di, ci, do, co, n_dims_edge)

    def udf_u_mul_e(self, do):
        """Calculate edge feature of a single output degree.

        Args:
        * do: degree of output features

        Returns:
        * fn: edge function handle
        """

        def fn(edges):
            # aggregate information from neighboring nodes
            msg = 0
            for ci, di in self.f_in.struct_list:
                src = edges.src['%d' % di].view(-1, ci * (2 * di + 1), 1)
                edge = edges.data['%d-%d' % (di, do)]
                msg = msg + torch.matmul(edge, src)
            msg = msg.view(msg.shape[0], -1, 2 * do + 1)  # -1 should be <co>

            return {'out-%d' % do: msg.view(msg.shape[0], -1, 2 * do + 1)}  # -1 should be <co>

        return fn

    def forward(self, graph, feat_dict, basis_dict, radial):
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * feat_dict: dict of node features
        * basis_dict: dict of equivariant bases, indexed by (d_i, d_o)
        * radial: radial distance of each edge in the graph

        Returns:
        * outputs: dict of output edge features
        """

        with graph.local_scope():
            # add node features to the local graph scope
            for key, val in feat_dict.items():
                graph.ndata[key] = val

            # compute edge features w/ PairwiseConv, and then add them to the local graph scope
            edge_feats = torch.cat([graph.edata['w'], radial], dim=-1)
            for di in self.f_in.degrees:
                for do in self.f_out.degrees:
                    graph.edata['%d-%d' % (di, do)] = \
                        self.pconv_dict['%d-%d' % (di, do)](edge_feats, basis_dict)

            # update edge features
            for do in self.f_out.degrees:
                graph.apply_edges(self.udf_u_mul_e(do))

            # gather edge features as outputs
            outputs = {'%d' % x: graph.edata['out-%d' % x] for x in self.f_out.degrees}

            return outputs

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'GConvSE3Partial: ' + ', '.join([
            'f_in={%s}' % str(self.f_in),
            'f_out={%s}' % str(self.f_out),
            'n_dims_edge=%d' % self.n_dims_edge,
        ])

        return repr_str
