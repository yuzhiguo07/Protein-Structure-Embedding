"""SE(3)-equivariant convolution between two single-type features."""

import logging

import torch
from torch import nn

from tfold_se3.utils import get_tensor_size
from tfold_se3.utils import get_peak_memory
from tfold_se3.modules.se3_trans.radial_func import RadialFunc


class PairwiseConv(nn.Module):
    """SE(3)-equivariant convolution between two single-type features."""

    def __init__(self, d_in, n_chns_in, d_out, n_chns_out, n_dims_edge=0):
        """SE(3)-equivariant convolution between a pair of feature types.

        Args:
        * d_in: input features' degree
        * n_chns_in: number of channels of input features
        * d_out: output features' degree
        * n_chns_out: number of channels of output features
        * n_dims_edge: number of dimensions for edge embeddings

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.d_in = d_in
        self.n_chns_in = n_chns_in
        self.d_out = d_out
        self.n_chns_out = n_chns_out
        self.n_dims_edge = n_dims_edge
        self.n_freqs = 2 * min(d_in, d_out) + 1

        # NN-parameterized radial function
        self.radial_fn = RadialFunc(self.n_freqs, self.n_chns_in, self.n_chns_out, self.n_dims_edge)

    def forward(self, inputs, basis_dict):
        """Perform the forward pass.

        Args:
        * inputs: input radial features of size BS x (D_e + 1)
        * basis_dict: dict of equivariant bases, indexed by (d_i, d_o)

        Returns:
        * outputs: output features of size BS x (C_o * (2 * d_o + 1)) x (C_i * (2 * d_i + 1))
        """

        # define abbreviations
        di = self.d_in
        do = self.d_out
        ci = self.n_chns_in
        co = self.n_chns_out
        di_e = 2 * di + 1
        do_e = 2 * do + 1
        nf = self.n_freqs  # which equals to 2 * min(di, do) + 1

        # perform the forward pass
        radial = self.radial_fn(inputs).view(-1, co * ci, nf)
        basis = basis_dict[(self.d_in, self.d_out)].view(-1, do_e * di_e, nf).permute(0, 2, 1)
        try:
            outputs = torch.bmm(radial, basis).view(-1, co, ci, do_e, di_e).permute(
                0, 1, 3, 2, 4).contiguous().view(-1, co * do_e, ci * di_e)
        except:
            logging.error('radial: %s / %.2f MB', str(radial.shape), get_tensor_size(radial))
            logging.error('basis: %s / %.2f MB', str(basis.shape), get_tensor_size(basis))
            logging.error('GPU memory (peak): %.2f MB', get_peak_memory())
            raise

        return outputs

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'PairwiseConv: ' + ', '.join([
            'd_in=%d' % self.d_in,
            'n_chns_in=%d' % self.n_chns_in,
            'd_out=%d' % self.d_out,
            'n_chns_out=%d' % self.n_chns_out,
            'n_dims_edge=%d' % self.n_dims_edge,
            'n_freqs=%d' % self.n_freqs,
        ])

        return repr_str
