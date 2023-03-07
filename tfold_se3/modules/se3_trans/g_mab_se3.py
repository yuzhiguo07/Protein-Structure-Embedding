"""SE(3)-equivariant multi-headed attention block.

This module compute a dict of node features via multi-headed attention, taking SE(3)-equivarient
  query, key, and value embeddings as inputs.
"""

import numpy as np
import torch
from torch import nn
import dgl
from dgl.nn.pytorch.softmax import edge_softmax

from tfold_se3.modules.se3_trans.fiber import fiber2head


class GMABSE3(nn.Module):
    """SE(3)-equivariant multi-headed attention block."""

    def __init__(self, f_key, f_val, n_heads):
        """Constructor function.

        Args:
        * f_key: fiber for key embeddings (same as the fiber for query embeddings)
        * f_val: fiber for value embeddings
        * n_heads: number of heads in the multi-headed attention

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.f_key = f_key
        self.f_val = f_val
        self.n_heads = n_heads

    @classmethod
    def udf_u_mul_e(cls, do):
        """Calculate edge feature of a single output degree.

        Args:
        * do: degree of output features

        Returns:
        * fn: edge function handle
        """

        def fn(edges):
            attn = edges.data['a']
            val = edges.data['v-%d' % do]
            msg = torch.reshape(attn, [*attn.shape, 1, 1]) * val

            return {'m': msg}

        return fn

    def forward(self, graph, qry_dict, key_dict, val_dict):
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * qry_dict: dict of per-node query embeddings
        * key_dict: dict of per-edge key embeddings
        * val_dict: dict of per-edge value embeddings

        Returns:
        * outputs: dict of output node features
        """

        with graph.local_scope():
            # add node/edge features to local graph scope
            for cv, dv in self.f_val.struct_list:
                assert cv % self.n_heads == 0, 'cv = %d / n_heads = %d' % (cv, self.n_heads)
                graph.edata['v-%d' % dv] = \
                    val_dict['%d' % dv].view(-1, self.n_heads, cv // self.n_heads, 2 * dv + 1)
            graph.edata['k'] = fiber2head(key_dict, self.n_heads, self.f_key)
            graph.ndata['q'] = fiber2head(qry_dict, self.n_heads, self.f_key)

            # compute multi-headed attention weights
            graph.apply_edges(dgl.function.e_dot_v('k', 'q', 'e'))
            graph.edata['a'] = edge_softmax(
                graph, graph.edata['e'] / np.sqrt(self.f_key.n_dims)).view(-1, self.n_heads)

            # update node features via attention-weighted message-passing
            for dv in self.f_val.degrees:
                graph.update_all(self.udf_u_mul_e(dv), dgl.function.sum('m', 'out-%d' % dv))

            # gather node features as outputs
            outputs = {}
            for cv, dv in self.f_val.struct_list:
                outputs['%d' % dv] = graph.ndata['out-%d' % dv].view(-1, cv, 2 * dv + 1)

            return outputs

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'GMABSE3: ' + ', '.join([
            'f_key={%s}' % str(self.f_key),
            'f_val={%s}' % str(self.f_val),
            'n_heads=%d' % self.n_heads,
        ])

        return repr_str
