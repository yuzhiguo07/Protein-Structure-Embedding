"""SE(3)-equivariant average pooling.

This module performs average pooling to aggregate node features into graph features.
"""

from torch import nn
from dgl.nn.pytorch.glob import AvgPooling


class GAvgPoolSE3(nn.Module):
    """SE(3)-equivariant average pooling."""

    def __init__(self):
        """Constructor function."""

        super().__init__()
        self.pool = AvgPooling()

    def forward(self, graph, feat_dict):
        """Perform average pooling on degree-0 features.

        Args:
        * graph: DGL graph
        * feat_dict: dict of node features, indexed by the degree

        Returns:
        * outputs: graph features aggregated from type-0 node features
        """

        node_feats = feat_dict['0'][..., 0]  # use type-0 features, and remove the last dimension
        graph_feats = self.pool(graph, node_feats)

        return graph_feats
