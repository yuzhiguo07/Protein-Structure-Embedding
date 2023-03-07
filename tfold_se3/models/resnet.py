"""Residual network for 2-D distance matrices of proteins."""

from torch import nn
from torch.nn.utils import spectral_norm


def get_conv2d(n_chns_in, n_chns_out, sp_norm, **kwargs):
    """Get a 2D convlutional layer, with spectral normalization enabled on demand."""

    conv = nn.Conv2d(n_chns_in, n_chns_out, **kwargs)
    if sp_norm:
        conv = spectral_norm(conv)

    return conv


def get_batchnorm2d(n_chns, sp_norm, **kwargs):
    """Get a 2D batch normalization layer, with spectral normalization enabled on demand."""

    batch_norm = nn.BatchNorm2d(n_chns, **kwargs)
    if sp_norm:
        batch_norm = spectral_norm(batch_norm)

    return batch_norm


def get_linear(n_dims_in, n_dims_out, sp_norm, **kwargs):
    """Get a linear layer, with spectral normalization enabled on demand."""

    linear = nn.Linear(n_dims_in, n_dims_out, **kwargs)
    if sp_norm:
        linear = spectral_norm(linear)

    return linear


class BasicBlock(nn.Module):
    """Basic block in a residual network."""

    def __init__(self, n_chns, sp_norm=False, gp_norm=False):
        """Constructor function."""

        super().__init__()

        # residue layers
        conv1 = get_conv2d(n_chns, n_chns, sp_norm, kernel_size=3, padding=1)
        norm1 = nn.GroupNorm(4, n_chns) if gp_norm else get_batchnorm2d(n_chns, sp_norm)
        conv2 = get_conv2d(n_chns, n_chns, sp_norm, kernel_size=3, padding=1)
        norm2 = nn.GroupNorm(4, n_chns) if gp_norm else get_batchnorm2d(n_chns, sp_norm)

        # pack all the layers into a network
        self.network = nn.Sequential(
            conv1,
            norm1,
            nn.ReLU(),
            conv2,
            norm2,
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Perform the forward pass."""

        return self.network(inputs) + inputs


class BottleneckBlock(nn.Module):
    """Bottleneck block in a residual network."""

    def __init__(self, n_chns_base, dilation, sp_norm=False, gp_norm=False):
        """Constructor function."""

        super().__init__()

        # validate hyper-parameters
        assert n_chns_base % 4 == 0, '# of channels must be a multiplier of 4'
        n_chns_shrk = n_chns_base // 4

        # residue layers
        conv1 = get_conv2d(n_chns_base, n_chns_shrk, sp_norm, kernel_size=1)
        norm1 = nn.GroupNorm(4, n_chns_shrk) if gp_norm else get_batchnorm2d(n_chns_shrk, sp_norm)
        conv2 = get_conv2d(
            n_chns_shrk, n_chns_shrk, sp_norm, kernel_size=3, padding=dilation, dilation=dilation)
        norm2 = nn.GroupNorm(4, n_chns_shrk) if gp_norm else get_batchnorm2d(n_chns_shrk, sp_norm)
        conv3 = get_conv2d(n_chns_shrk, n_chns_base, sp_norm, kernel_size=1)
        norm3 = nn.GroupNorm(4, n_chns_base) if gp_norm else get_batchnorm2d(n_chns_base, sp_norm)

        # pack all the layers into a network
        self.network = nn.Sequential(
            conv1,
            norm1,
            nn.ReLU(),
            conv2,
            norm2,
            nn.ReLU(),
            conv3,
            norm3,
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Perform the forward pass."""

        return self.network(inputs) + inputs


class Resnet(nn.Module):
    """Residual network.

    * Architecture:
      * input conv
      * residual blocks
      * (optional) adaptive pooling
      * output conv

    * NCE-based EBM training:
      * inputs: N x C_i x L_i x L_i
      * outputs: N x 1
      * hyper-parameters:
        * n_chns_out: 1
        * enbl_pool: True

    * DSM-based EBM training:
      * inputs: N x C_i x L_i x L_i
      * outputs: N x C_o x L_i x L_i (normally, we have C_o = 3)
      * hyper-parameters:
        * enbl_pool: False
    """

    def __init__(self,
                 n_chns_in,
                 n_chns_out,
                 n_blocks,
                 n_chns_hid=16,
                 enbl_pool=True,
                 n_pool_grids=1,
                 block_type='basic',
                 sp_norm=False,
                 gp_norm=False):
        """Constructor function.

        Args:
        * n_chns_in: number of input feature maps' channels
        * n_chns_out: number of output feature maps' channels
        * n_blocks: number of residual blocks
        * n_chns_hid: number of hidden feature maps' channels
        * enbl_pool: whether to enable the global pooling operation
        * n_pool_grids: number of grids (per direction) after global pooling
        * block_type: residual block's type ('basic' OR 'bottleneck')
        * sp_norm: whether to enable spectral normalization on Conv2D / BatchNorm2d / Linear layers
        * gp_norm: whether to use GroupNorm instead of BatchNorm

        Returns: n/a

        Notes:
        * If <enbl_pool> is disabled, then outputs will be of size N x C_o x L x L; otherwise,
            outputs will be of size N x C_o x G x G, where N is the batch size, C_o is the number
            of output feature maps' channels, L is the sequence length, and G is the number of grids
            (per direction) after global pooling.
        """

        super().__init__()

        # input block
        block_in = nn.Sequential(
            get_conv2d(n_chns_in, n_chns_hid, sp_norm, kernel_size=3, padding=1),
            nn.GroupNorm(4, n_chns_hid) if sp_norm else get_batchnorm2d(n_chns_hid, sp_norm),
            nn.ReLU(),
        )

        # hidden block
        blocks_hid = self.__build_blocks_hid(n_chns_hid, n_blocks, block_type, sp_norm, gp_norm)

        # output block
        if not enbl_pool:
            block_out = nn.Sequential(
                get_conv2d(n_chns_hid, n_chns_out, sp_norm, kernel_size=3, padding=1),
            )
        else:
            block_out = nn.Sequential(
                nn.AdaptiveAvgPool2d((n_pool_grids, n_pool_grids)),
                nn.Flatten(),
                nn.ReLU(),
                get_linear(n_chns_hid * n_pool_grids * n_pool_grids, n_chns_out, sp_norm),
            )

        # pack all the layers into a network
        self.network = nn.Sequential(
            block_in,
            *blocks_hid,
            block_out,
        )

    def forward(self, inputs):
        """Perform the forward pass."""

        return self.network(inputs)

    @classmethod
    def __build_blocks_hid(cls, n_chns, n_blocks, block_type, sp_norm, gp_norm):
        """Build residual blocks for hidden layers."""

        # configurations
        dilation_list = [1, 3, 5, 9]
        n_dilations = len(dilation_list)

        # validate hyper-parameters
        assert n_blocks % n_dilations == 0, '# of blocks must be a multiplier of %d' % n_dilations
        assert block_type in ['basic', 'bottleneck'], 'unrecognized block type: ' + block_type

        # build residual blocks for hidden layers
        blocks = [None for _ in range(n_blocks)]
        for idx in range(n_blocks):
            dilation = dilation_list[idx % n_dilations]
            if block_type == 'basic':
                blocks[idx] = BasicBlock(n_chns, sp_norm, gp_norm)
            else:
                blocks[idx] = BottleneckBlock(n_chns, dilation, sp_norm, gp_norm)

        return blocks

def main():
    """Main entry."""

    # configurations
    n_chns_in = 16
    n_chns_out = 4
    n_blocks = 8

    # create a residual network
    model = Resnet(n_chns_in, n_chns_out, n_blocks)
    print(model.network)

if __name__ == '__main__':
    main()
