"""SE(3)-equivariant max pooling.

This module performs max pooling to aggregate node features into graph features.
"""

from torch import nn
from dgl.nn.pytorch.glob import MaxPooling


class GMaxPoolSE3(nn.Module):
    """SE(3)-equivariant max pooling."""

    def __init__(self):
        """Constructor function."""

        super().__init__()
        self.pool = MaxPooling()

    def forward(self, graph, feat_dict):
        """Perform max pooling on degree-0 features.

        Args:
        * graph: DGL graph
        * feat_dict: dict of node features, indexed by the degree

        Returns:
        * outputs: graph features aggregated from type-0 node features
        """

        node_feats = feat_dict['0'][..., 0]  # use type-0 features, and remove the last dimension
        graph_feats = self.pool(graph, node_feats)

        return graph_feats
