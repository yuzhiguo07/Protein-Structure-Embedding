"""Unit-tests for SE(3)-equivariant modules."""

from torch import nn

from tfold_se3.modules.se3_trans import Fiber
from tfold_se3.modules.se3_trans import GBatchNormSE3
from tfold_se3.modules.se3_trans import GCondBatchNormSE3
from tfold_se3.modules.se3_trans import GConvSE3
from tfold_se3.modules.se3_trans import GResSE3
from tfold_se3.modules.se3_trans import GNormSE3
from tfold_se3.utils import check_se3_equiv


def test_g_batch_norm_se3(config):
    """Test the <GBatchNormSE3> module."""

    # define the forward pass function
    def _forward_fn(module, _0, feat_dict, _2, _3, _4):
        return module(feat_dict)

    # build a <GBatchNormSE3> module
    fiber = Fiber(n_degrees=2, n_channels=config['n_dims_node'])
    module = GBatchNormSE3(fiber)

    # check the SE(3)-equivariance
    check_se3_equiv(module, config, forward_fn=_forward_fn)


def test_g_cond_batch_norm_se3(config):
    """Test the <GCondBatchNormSE3> module."""

    # define the forward pass function
    def _forward_fn(module, _0, feat_dict, _2, _3, cond_idxs):
        return module(feat_dict, cond_idxs)

    # build a <GCondBatchNormSE3> module
    fiber = Fiber(n_degrees=2, n_channels=config['n_dims_node'])
    module = GCondBatchNormSE3(fiber, cond_depth=config['cond_depth'])

    # check the SE(3)-equivariance
    check_se3_equiv(module, config, forward_fn=_forward_fn)


def test_g_conv_se3(config):
    """Test the <GConvSE3> module."""

    # define the forward pass function
    def _forward_fn(module, graph, feat_dict, basis_dict, radial, _4):
        return module(graph, feat_dict, basis_dict, radial)

    # build a <GConvSE3> module
    f_in = Fiber(n_degrees=2, n_channels=config['n_dims_node'])
    f_out = Fiber(n_degrees=2, n_channels=config['n_dims_node'])
    module = GConvSE3(f_in, f_out, self_interact=True, n_dims_edge=config['n_dims_edge'])

    # check the SE(3)-equivariance
    check_se3_equiv(module, config, forward_fn=_forward_fn)


def test_g_res_se3(config):
    """Test the <GResSE3> module."""

    # configurations
    div_fctr = 4
    n_heads = 2

    # define the forward pass function
    def _forward_fn(module, graph, feat_dict, basis_dict, radial, _4):
        return module(graph, feat_dict, basis_dict, radial)

    # build a <GResSE3> module
    f_in = Fiber(n_degrees=2, n_channels=config['n_dims_node'])
    f_out = Fiber(n_degrees=2, n_channels=config['n_dims_node'])
    module = GResSE3(
        f_in, f_out, n_dims_edge=config['n_dims_edge'], div_fctr=div_fctr, n_heads=n_heads)

    # check the SE(3)-equivariance
    check_se3_equiv(module, config, forward_fn=_forward_fn)


def test_se3_trans(config):
    """Test the SE(3)-Transformer model."""

    # configurations
    n_chns_hid = 16
    n_blks = 4
    div_fctr = 4
    n_heads = 2

    # define the forward pass function
    def _forward_fn(model, graph, feat_dict, basis_dict, radial, _4):
        node_feats = feat_dict
        for layer in model:
            if isinstance(layer, GNormSE3):
                node_feats = layer(node_feats)
            else:
                node_feats = layer(graph, node_feats, basis_dict, radial)
        return node_feats

    # setup input / hidden / output fibers
    f_in = Fiber(n_degrees=2, n_channels=config['n_dims_node'])
    f_hid = Fiber(n_degrees=(config['max_degree'] + 1), n_channels=n_chns_hid)
    f_out = Fiber(n_degrees=2, n_channels=config['n_dims_node'])

    # build a SE(3)-Transformer model
    network = nn.ModuleList()
    for idx_blk in range(n_blks):
        f_src = f_in if idx_blk == 0 else f_hid
        network.append(GResSE3(
            f_src, f_hid, n_dims_edge=config['n_dims_edge'], div_fctr=div_fctr, n_heads=n_heads))
        network.append(GNormSE3(f_hid))
    network.append(GConvSE3(f_hid, f_out, self_interact=True, n_dims_edge=config['n_dims_edge']))

    # check the SE(3)-equivariance
    check_se3_equiv(network, config, forward_fn=_forward_fn)


def main():
    """Main entry."""

    # configurations
    config = {
        'n_nodes': 32,
        'max_degree': 1,  # type-0 & type-1 features are used
        'n_dims_node': 8,
        'n_dims_edge': 12,
        'cond_depth': 16,
    }

    # test the <GBatchNormSE3> module
    test_g_batch_norm_se3(config)

    # test the <GCondBatchNormSE3> module
    test_g_cond_batch_norm_se3(config)

    # test the <GConvSE3> module
    test_g_conv_se3(config)

    # test the <GResSE3> module
    test_g_res_se3(config)

    # test the SE(3)-Transformer model
    test_se3_trans(config)

if __name__ == '__main__':
    main()
