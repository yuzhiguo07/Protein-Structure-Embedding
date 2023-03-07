"""Neural network parameterized radial function."""

from torch import nn


class RadialFunc(nn.Module):
    """Neural network parameterized radial function.

    TODO:
    * check whether LayerNorm is the optimal choice
    """

    def __init__(self, n_freqs, n_chns_in, n_chns_out, n_dims_edge=0, n_dims_hid=32):
        """Constructor function.

        Args:
        * n_freqs: number of output frequencies
        * n_chns_in: number of channels of input features
        * n_chns_out: number of channels of output features
        * n_dims_edge: number of dimensions for edge embeddings
        * n_dims_hid: number of dimension for hidden layer activations

        Returns: n/a
        """

        # initialization
        super().__init__()
        self.n_freqs = n_freqs
        self.n_chns_in = n_chns_in
        self.n_chns_out = n_chns_out
        self.n_dims_edge = n_dims_edge
        self.n_dims_hid = n_dims_hid

        # build a MLP network
        self.net = nn.Sequential(
            #nn.BatchNorm1d(self.n_dims_edge + 11),
            nn.Linear(self.n_dims_edge + 11, self.n_dims_hid),
            nn.LayerNorm(self.n_dims_hid),
            #nn.BatchNorm1d(self.n_dims_hid),
            nn.ReLU(),
            nn.Linear(self.n_dims_hid, self.n_dims_hid),
            nn.LayerNorm(self.n_dims_hid),
            #nn.BatchNorm1d(self.n_dims_hid),
            nn.ReLU(),
            nn.Linear(self.n_dims_hid, self.n_chns_out * self.n_chns_in * self.n_freqs),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
        #nn.init.kaiming_uniform_(self.net[0].weight)
        #nn.init.kaiming_uniform_(self.net[3].weight)
        #nn.init.kaiming_uniform_(self.net[6].weight)

    def forward(self, inputs):
        """Perform the forward pass.

        Args:
        * inputs: input radial features of size BS x (D_e + 1)

        Returns:
        * outputs: output features of size BS x C_o x 1 x C_i x 1 x NF
        """

        outputs = self.net(inputs).view(-1, self.n_chns_out, 1, self.n_chns_in, 1, self.n_freqs)

        return outputs

    def __repr__(self):
        """Get the string representation."""

        repr_str = 'RadialFunc: ' + ', '.join([
            'n_freqs=%d' % self.n_freqs,
            'n_chns_in=%d' % self.n_chns_in,
            'n_chns_out=%d' % self.n_chns_out,
            'n_dims_edge=%d' % self.n_dims_edge,
            'n_dims_hid=%d' % self.n_dims_hid,
        ])

        return repr_str
