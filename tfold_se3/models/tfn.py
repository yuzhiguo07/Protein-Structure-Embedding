# pylint: skip-file

import sys
import logging
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from dgl.nn.pytorch import NNConv
from dgl.nn.pytorch import GraphConv

from tfold_se3.from_se3trans.fibers import Fiber
from tfold_se3.from_se3trans.modules import GSE3Res
from tfold_se3.from_se3trans.modules import GConvSE3
from tfold_se3.from_se3trans.modules import GNormSE3
from tfold_se3.from_se3trans.modules import GMaxPooling
from tfold_se3.from_se3trans.modules import GAvgPooling
from tfold_se3.from_se3trans.modules import get_basis_and_r


class TFN(nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int, atom_feature_size: int,
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4,
                 edge_dim: int=4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels*num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks

        logging.info('Block #0:\n{}'.format(self.block0))
        logging.info('Block #1:\n{}'.format(self.block1))
        logging.info('Block #2:\n{}'.format(self.block2))

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers-1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(block0), nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        return h
