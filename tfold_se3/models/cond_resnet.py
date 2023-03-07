"""Residual network with support for conditional normalization layers.

Customized layers:
* CondBatchNorm2d: (scale, bias) depends on the noise level
* CondInstanceNorm2d: (scale, bias) depends on the noise level
* CondInstanceNormPlus2d: (scale, bias) depends on the noise level, and an additional term to
    account for the shifted color issue (depends on the k-th feature map's mean value)
"""
import torch
from torch import nn

from tfold_se3.modules import CrissCrossAttention
from tfold_se3.modules import CondBatchNorm2d
from tfold_se3.modules import CondBatchNorm2dLegacy
from tfold_se3.modules import CondInstanceNorm2d


def get_norm_layer(n_chns, layer_type='batch_norm', layer_depth=None):
    """Get a normalization layer of specified type.

    Args:
    * n_chns: number of feature map channels
    * layer_type: normalization layer type
    * layer_depth: number of conditional classes / noise levels

    Returns:
    * layer: normalization layer
    """

    if layer_type == 'batch_norm':
        layer = nn.BatchNorm2d(n_chns)
    elif layer_type == 'instance_norm':
        layer = nn.InstanceNorm2d(n_chns)
    elif layer_type == 'cond_batch_norm':
        assert isinstance(layer_depth, int), '<layer_depth> must be an integer'
        layer = CondBatchNorm2d(n_chns, layer_depth, bias=True)
    elif layer_type == 'cond_batch_norm_legacy':
        assert isinstance(layer_depth, int), '<layer_depth> must be an integer'
        layer = CondBatchNorm2dLegacy(n_chns, layer_depth, bias=True)
    elif layer_type == 'cond_instance_norm':
        assert isinstance(layer_depth, int), '<layer_depth> must be an integer'
        layer = CondInstanceNorm2d(
            n_chns, layer_depth, bias=True, fix_color_shift=False, track_running_stats=False)
    elif layer_type == 'cond_instance_norm_plus':
        assert isinstance(layer_depth, int), '<layer_depth> must be an integer'
        layer = CondInstanceNorm2d(
            n_chns, layer_depth, bias=True, fix_color_shift=True, track_running_stats=False)
    elif layer_type == 'cond_instance_norm_trs':
        assert isinstance(layer_depth, int), '<layer_depth> must be an integer'
        layer = CondInstanceNorm2d(
            n_chns, layer_depth, bias=True, fix_color_shift=False, track_running_stats=True)
    elif layer_type == 'cond_instance_norm_plus_trs':
        assert isinstance(layer_depth, int), '<layer_depth> must be an integer'
        layer = CondInstanceNorm2d(
            n_chns, layer_depth, bias=True, fix_color_shift=True, track_running_stats=True)
    else:
        raise NotImplementedError('layer type <%s> is not implemented' % layer_type)

    return layer


class BasicBlock(nn.Module):
    """Basic block in a residual network.

    Architecture: full pre-activation
    Ref: He et al., Identity Mappings in Deep Residual Networks. ECCV 2016. - Fig. 4(e)
    """

    def __init__(self, n_chns, dilation, norm_layer_type='batch_norm', norm_layer_depth=None):
        """Constructor function."""

        super().__init__()

        # basic configurations
        self.has_cond = norm_layer_type.startswith('cond_')

        # build layers within the residual block
        self.actv = nn.ELU()
        self.norm1 = get_norm_layer(n_chns, norm_layer_type, norm_layer_depth)
        self.conv1 = nn.Conv2d(n_chns, n_chns, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = get_norm_layer(n_chns, norm_layer_type, norm_layer_depth)
        self.conv2 = nn.Conv2d(n_chns, n_chns, kernel_size=3, padding=1)

        # initialize all the convolutional layers' weights
        self.conv1.weight.data.normal_(0.0, 1e-3)
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(0.0, 1e-3)
        self.conv2.bias.data.zero_()

    def forward(self, inputs, idxs):
        """Perform the forward pass."""

        outputs = self.norm1(inputs, idxs) if self.has_cond else self.norm1(inputs)
        outputs = self.actv(outputs)
        outputs = self.conv1(outputs)

        outputs = self.norm2(outputs, idxs) if self.has_cond else self.norm2(outputs)
        outputs = self.actv(outputs)
        outputs = self.conv2(outputs)

        return inputs + outputs


class BottleneckBlock(nn.Module):
    """Bottleneck block in a residual network.

    Architecture: full pre-activation
    Ref: He et al., Identity Mappings in Deep Residual Networks. ECCV 2016. - Fig. 4(e)
    """

    def __init__(self, n_chns_base, dilation, norm_layer_type='batch_norm', norm_layer_depth=None):
        """Constructor function."""

        super().__init__()

        # basic configurations
        self.has_cond = norm_layer_type.startswith('cond_')

        # validate hyper-parameters
        assert n_chns_base % 4 == 0, '# of channels must be a multiplier of 4'
        n_chns_shrk = n_chns_base // 4

        # build layers within the residual block
        self.actv = nn.ELU()
        self.norm1 = get_norm_layer(n_chns_base, norm_layer_type, norm_layer_depth)
        self.conv1 = nn.Conv2d(n_chns_base, n_chns_shrk, kernel_size=1)
        self.norm2 = get_norm_layer(n_chns_shrk, norm_layer_type, norm_layer_depth)
        self.conv2 = nn.Conv2d(
            n_chns_shrk, n_chns_shrk, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm3 = get_norm_layer(n_chns_shrk, norm_layer_type, norm_layer_depth)
        self.conv3 = nn.Conv2d(n_chns_shrk, n_chns_base, kernel_size=1)

        # initialize all the convolutional layers' weights
        self.conv1.weight.data.normal_(0.0, 1e-3)
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.normal_(0.0, 1e-3)
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.normal_(0.0, 1e-3)
        self.conv3.bias.data.zero_()

    def forward(self, inputs, idxs):
        """Perform the forward pass."""

        outputs = self.norm1(inputs, idxs) if self.has_cond else self.norm1(inputs)
        outputs = self.actv(outputs)
        outputs = self.conv1(outputs)

        outputs = self.norm2(outputs, idxs) if self.has_cond else self.norm2(outputs)
        outputs = self.actv(outputs)
        outputs = self.conv2(outputs)

        outputs = self.norm3(outputs, idxs) if self.has_cond else self.norm3(outputs)
        outputs = self.actv(outputs)
        outputs = self.conv3(outputs)

        return inputs + outputs

class Prednet(nn.Module):
    def __init__(self,
                 n_chns_in=16,
                 n_chns_hid=32, 
                 n_cls=10):
        super().__init__()
        self.pred_layers = nn.Sequential(
            nn.Linear(n_chns_in, n_chns_hid),
            nn.ReLU(),
            nn.Linear(n_chns_hid, n_cls),
        )


    def forward(self, encoder_feat):
        avg_pool = nn.AvgPool2d(encoder_feat.shape[-1], stride=2)
        encoder_feat = avg_pool(encoder_feat)
        encoder_feat = torch.squeeze(encoder_feat, 3)
        encoder_feat = torch.squeeze(encoder_feat, 2)       
        pred_out = self.pred_layers(encoder_feat)
        return pred_out


class CondResnet(nn.Module):
    """Residual network with support for conditional normalization layers."""

    def __init__(self,
                 n_chns_in,
                 n_chns_out,
                 n_blocks,
                 n_chns_hid=16,
                 block_type='basic',
                 norm_layer_type='batch_norm',
                 norm_layer_depth=None,
                 use_cc_attn=False):
        """Constructor function.

        Args:
        * n_chns_in: number of input feature maps' channels
        * n_chns_out: number of output feature maps' channels
        * n_blocks: number of residual blocks
        * n_chns_hid: number of hidden feature maps' channels
        * block_type: residual block's type ('basic' OR 'bottleneck')
        * norm_layer_type: normalization layer's type
        * norm_layer_depth: number of conditional classes / noise levels

        Returns: n/a
        """

        super().__init__()

        # configurations
        dilation_list = [1, 3, 5, 9]
        n_dilations = len(dilation_list)

        # validate hyper-parameters
        assert n_blocks % n_dilations == 0, '# of blocks must be a multiplier of %d' % n_dilations
        assert block_type in ['basic', 'bottleneck'], 'unrecognized block type: ' + block_type

        # basic configurations
        self.has_cond = norm_layer_type.startswith('cond_')

        # input block
        self.conv_in = nn.Conv2d(n_chns_in, n_chns_hid, kernel_size=3, padding=1)

        # hidden blocks
        self.blocks_hid = nn.ModuleList()
        for idx in range(n_blocks):
            dilation = dilation_list[idx % n_dilations]
            if block_type == 'basic':
                block = BasicBlock(n_chns_hid, dilation, norm_layer_type, norm_layer_depth)
            else:
                block = BottleneckBlock(n_chns_hid, dilation, norm_layer_type, norm_layer_depth)
            self.blocks_hid.append(block)
            if use_cc_attn and dilation == dilation_list[-1]:
                self.blocks_hid.append(CrissCrossAttention(n_chns_hid))

        # output block
        self.actv = nn.ELU()
        self.norm_out = get_norm_layer(n_chns_hid, norm_layer_type, norm_layer_depth)
        self.conv_out = nn.Conv2d(n_chns_hid, n_chns_out, kernel_size=3, padding=1)

        # initialize all the convolutional layers' weights
        self.conv_in.weight.data.normal_(0.0, 1e-3)
        self.conv_in.bias.data.zero_()
        self.conv_out.weight.data.normal_(0.0, 1e-3)
        self.conv_out.bias.data.zero_()

    def forward(self, inputs, idxs):
        """Perform the forward pass."""

        # input block
        outputs = self.conv_in(inputs)

        # hidden blocks
        for block in self.blocks_hid:
            if isinstance(block, (BasicBlock, BottleneckBlock)):
                outputs = block(outputs, idxs)
            else:
                outputs = block(outputs)

        # output block
        outputs = self.norm_out(outputs, idxs) if self.has_cond else self.norm_out(outputs)
        encoder_feat = self.actv(outputs)
        outputs = self.conv_out(encoder_feat)

        return outputs, encoder_feat


def main():
    """Main entry."""

    # configurations
    n_chns_in = 16
    n_chns_out = 4
    n_blocks = 8
    n_chns_hid = 32
    block_type = 'basic'
    norm_layer_type = 'cond_instance_norm_plus'
    norm_layer_depth = 16

    # create a residual network
    model = CondResnet(
        n_chns_in, n_chns_out, n_blocks, n_chns_hid, block_type, norm_layer_type, norm_layer_depth)
    print(model)

if __name__ == '__main__':
    main()
