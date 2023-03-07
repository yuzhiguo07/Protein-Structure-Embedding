"""SE(3)-equivariance related utility functions."""

import numpy as np
from scipy.spatial.distance import cdist
import dgl
import torch

from tfold_se3.utils.math_utils import get_rotate_mat
from tfold_se3.from_se3cnn import utils_steerable


def get_basis(sh_dict, max_degree):
    """Get SE(3)-equivariant bases for all the (d_i, d_o) combinations.

    Args:
    * sh_dict: dict of spherical harmonics, indexed by <d>
    * max_degree: maximal degree of feature types

    Returns:
    * basis_dict: dict of equivariant bases, indexed by (d_i, d_o)

    Note:
    * {d, d_i, d_o} take values from 0 to <max_degree>, inclusively.
    * each basis is of size BS x 1 x (2 * d_o + 1) x 1 x (2 * d_i + 1) * (2 * min(d_i, d_o) + 1)
    """

    basis_dict = {}
    device = sh_dict[0].device
    with torch.no_grad():
        for d_i in range(max_degree + 1):
            for d_o in range(max_degree + 1):
                K_Js = []
                for J in range(abs(d_i - d_o), d_i + d_o + 1):
                    Q_J = utils_steerable.basis_transformation_Q_J(J, d_i, d_o)
                    K_J = torch.matmul(sh_dict[J], Q_J.float().T.to(device))
                    K_Js.append(K_J)
                basis_shape = (-1, 1, 2 * d_o + 1, 1, 2 * d_i + 1, 2 * min(d_i, d_o) + 1)
                basis_dict[(d_i, d_o)] = torch.stack(K_Js, dim=-1).view(basis_shape)

    return basis_dict


def get_basis_and_radial(graph, max_degree):
    """Get SE(3)-equivariant bases and radial distance.

    Args:
    * graph: DGL graph
    * max_degree: maximal degree of feature types

    Returns:
    * basis_dict: dict of equivariant bases, indexed by (d_i, d_o)
    * radial: radial distance of each edge in the graph
    """

    r_ij = utils_steerable.get_spherical_from_cartesian_torch(graph.edata['d'])
    sh_dict = utils_steerable.precompute_sh(r_ij, 2 * max_degree)
    basis_dict = get_basis(sh_dict, max_degree)
    radial = torch.sqrt(torch.sum(graph.edata['d'] ** 2, -1, keepdim=True))

    return basis_dict, radial


def build_graph(n_nodes, n_dims_node, n_dims_edge):
    """Build a graph with randomized 3D coordinates & node/edge features.

    Args:
    * n_nodes: number of nodes
    * n_dims_node: number of dimensions of input node features
    * n_dims_edge: number of dimensions of input edge features

    Returns:
    * graph: DGL graph
    """

    # randomly generate 3D coordinates and node/edge features
    cord_mat = np.random.normal(size=(n_nodes, 3)).astype(np.float32)
    node_feats = np.random.normal(size=(n_nodes, n_dims_node)).astype(np.float32)
    dist_mat = cdist(cord_mat, cord_mat, metric='euclidean')
    dist_thres = np.percentile(dist_mat, 25)
    edge_idxs = np.nonzero((1 - np.eye(n_nodes)) * (dist_mat <= dist_thres))
    n_edges = edge_idxs[0].size
    dcrd_mat = cord_mat[edge_idxs[0]] - cord_mat[edge_idxs[1]]
    edge_feats = np.random.normal(size=(n_edges, n_dims_edge)).astype(np.float32)

    # create a DGL graph
    graph = dgl.DGLGraph()
    graph.add_nodes(n_nodes)
    graph.add_edges(edge_idxs[0], edge_idxs[1])
    graph.ndata['x'] = torch.tensor(cord_mat, dtype=torch.float32)
    graph.ndata['f'] = torch.tensor(node_feats[..., None], dtype=torch.float32)
    graph.edata['d'] = torch.tensor(dcrd_mat, dtype=torch.float32)
    graph.edata['w'] = torch.tensor(edge_feats, dtype=torch.float32)

    return graph

def build_addi_inputs(graph, config):
    """Build additional inputs for SE(3)-equivariant modules.

    Args:
    * graph: DGL graph
    * config: dict of configurations

    Returns:
    * feat_dict: dict of node features
    * basis_dict: dict of equivariant bases, indexed by (d_i, d_o)
    * radial: radial distance of each edge in the graph
    * cond_idxs: conditional indices of size N_v
    """

    feat_dict = {
        '0': graph.ndata['f'],
        '1': torch.unsqueeze(graph.ndata['x'], dim=1).repeat(1, config['n_dims_node'], 1),
    }
    basis_dict, radial = get_basis_and_radial(graph, config['max_degree'])
    cond_idxs = np.random.randint(config['cond_depth']) \
        * torch.ones((config['n_nodes']), dtype=torch.int64)

    return feat_dict, basis_dict, radial, cond_idxs


def update_graph(graph, rotate_mat):
    """Update the DGL graph with the specified 3D rotation.

    Args:
    * graph: DGL graph
    * rotate_mat: 3D rotation matrix

    Returns:
    * graph: DGL graph
    """

    cord_mat_old = graph.ndata['x']
    cord_mat_new = torch.matmul(cord_mat_old, torch.tensor(rotate_mat, dtype=torch.float32))
    idxs_u, idxs_v = graph.all_edges(form='uv', order='eid')
    graph.ndata['x'] = cord_mat_new
    graph.edata['d'] = cord_mat_new[idxs_u] - cord_mat_new[idxs_v]

    return graph


def check_se3_equiv(module, config, build_addi_inputs_fn=build_addi_inputs, forward_fn=None):
    """Check the SE(3)-equivariance of specified module.

    Args:
    * module: SE(3)-equivariant module to be tested
    * config: dict of configurations
    * build_addi_inputs_fn: customized function for building additional inputs
    * forward_fn: customized function for performing the forward pass

    Returns: n/a
    """

    # display the greeting message
    print('check the SE(3)-equivariance for ' + str(module))

    # build inputs from original 3D coordinates
    graph = build_graph(config['n_nodes'], config['n_dims_node'], config['n_dims_edge'])
    feat_dict, basis_dict, radial, cond_idxs = build_addi_inputs_fn(graph, config)

    # perform the forward pass w/ original 3D coordinates
    if forward_fn is None:
        outputs = module(graph, feat_dict, basis_dict, radial, cond_idxs)
    else:
        outputs = forward_fn(module, graph, feat_dict, basis_dict, radial, cond_idxs)
    cord_mat_src = outputs['1'].detach().cpu().numpy()[:, 0, :]

    # build inputs w/ a randomized 3D rotation; <cond_idxs> is kept the same
    rotate_mat = get_rotate_mat()
    graph = update_graph(graph, rotate_mat)
    feat_dict, basis_dict, radial, _ = build_addi_inputs_fn(graph, config)

    # perform the forward pass w/ transformed 3D coordinates
    if forward_fn is None:
        outputs = module(graph, feat_dict, basis_dict, radial, cond_idxs)
    else:
        outputs = forward_fn(module, graph, feat_dict, basis_dict, radial, cond_idxs)
    cord_mat_dst = outputs['1'].detach().cpu().numpy()[:, 0, :]

    # verify the SE(3)-equivariance
    cord_mat_ref = np.matmul(cord_mat_src, rotate_mat)
    print('D-S: %.4e' % np.mean(np.linalg.norm(cord_mat_dst - cord_mat_src, axis=-1)))
    print('D-R: %.4e' % np.mean(np.linalg.norm(cord_mat_dst - cord_mat_ref, axis=-1)))
