"""Dataset for EBM training & sampling - 2D & 3D inputs."""

import logging

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import cdist

from tfold_se3.utils.prof_utils import *  # pylint: disable=wildcard-import
from tfold_se3.utils import AA_NAMES_1CHAR
from tfold_se3.utils import cvt_to_one_hot
# from tfold_se3.datasets.cath_pdb import CathPdbDatasetConfig
# from tfold_se3.datasets.cath_pdb import CathPdbDataset
from tfold_se3.datasets.rcsb_pdb import RcsbPdbDatasetConfig
from tfold_se3.datasets.rcsb_pdb import RcsbPdbDataset


@profile
def get_crop_bnds(seq_len, crop_mode, crop_size, batch_size):
    """Get residue indices' boundaries for random cropping."""

    # get residue indices' boundaries for random cropping
    crop_bnds = np.zeros((batch_size, 2, 2), dtype=np.int32)
    if crop_mode == 'none':
        crop_bnds[:, :, 1] = seq_len
    elif crop_mode == 'single':
        crop_bnds[:, :, 0] = np.repeat(
            np.random.randint(seq_len - crop_size + 1, size=(1, 2)), batch_size, axis=0)
        crop_bnds[:, :, 1] = crop_bnds[:, :, 0] + crop_size
    else:  # then <crop_mode> must be 'multi'
        crop_bnds[:, :, 0] = np.random.randint(seq_len - crop_size + 1, size=(batch_size, 2))
        crop_bnds[:, :, 1] = crop_bnds[:, :, 0] + crop_size

    return crop_bnds


@profile
def get_denc_tns_prob(dist_tns, n_dims_denc):
    """Get distance encodings - PyTorch-based implementation.

    Args:
    * dist_tns: PyTorch tensor of size BS x H x W
    * n_dims: number of dimensions for distance encodings

    Returns:
    * dens_tns: PyTorch tensor of size BS x D x H x W
    """

    # configurations
    radius = 3.0
    epsilon = 1e-6
    device = dist_tns.device  # use the same device as <dist_tns>

    # determine anchor distance values
    if n_dims_denc == 24:
        dist_vals_np = np.concatenate([
            np.linspace(0.0, 20.0, 21),
            np.array([22.0, 25.0, 30.0]),
        ]).astype(np.float32)
    elif n_dims_denc == 61:
        dist_vals_np = np.concatenate([
            np.linspace(0.0, 30.0, 31),
            np.linspace(32.0, 70.0, 20),
            np.linspace(73.0, 100.0, 10),
        ]).astype(np.float32)
    else:
        raise ValueError('unsupported # of dimensions for distance encodings: %d' % n_dims_denc)

    # get distance encodings
    dist_vals = torch.tensor(dist_vals_np, device=device)
    dist_tns_clip = torch.clamp(dist_tns, np.min(dist_vals_np), np.max(dist_vals_np))
    diff_tns = torch.unsqueeze(dist_tns_clip, dim=1) - torch.reshape(dist_vals, [1, -1, 1, 1])
    scor_tns = torch.exp(-torch.square(diff_tns) / (2 * radius ** 2))
    denc_tns = scor_tns / (torch.sum(scor_tns, dim=1, keepdim=True) + epsilon)

    return denc_tns


@profile
def get_denc_tns_sigm(dist_tns, n_dims_denc):
    """Get distance encoding - sigmoid based.

    Args:
    * dist_tns: PyTorch tensor of size BS x H x W
    * n_dims: number of dimensions for distance encodings

    Returns:
    * dens_tns: PyTorch tensor of size BS x D x H x W
    """

    # configurations
    base = 2.0
    device = dist_tns.device  # use the same device as <dist_tns>

    # get distance encodings
    dist_vals_np = np.power(base, np.linspace(0.0, 10.0, num=n_dims_denc))[None, :, None, None]
    dist_vals = torch.tensor(dist_vals_np, dtype=torch.float32, device=device)
    denc_tns = torch.sigmoid(torch.unsqueeze(dist_tns, dim=1) / dist_vals - 1.0)

    return denc_tns


@profile
def find_edges_by_dthres(cord_mat, dist_thres, n_edges_max):
    """Find edges by the distance threshold."""

    # initialization
    seq_len = cord_mat.shape[0]

    # primary edges (adjacent residues)
    edge_idxs_pri = np.stack([
        np.concatenate([np.arange(seq_len - 1), np.arange(1, seq_len)]),
        np.concatenate([np.arange(1, seq_len), np.arange(seq_len - 1)]),
    ], axis=0)

    # secondary edges (CA-CA distance lower than the threshold)
    mask_mat = (np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]) > 1)
    dist_mat = cdist(cord_mat, cord_mat, metric='euclidean')
    edge_idxs_sec = np.array(np.nonzero(mask_mat * (dist_mat <= dist_thres)))
    if n_edges_max != -1 and edge_idxs_sec.shape[1] > n_edges_max:
        idxs = np.random.choice(edge_idxs_sec.shape[1], n_edges_max, replace=False)
        edge_idxs_sec = edge_idxs_sec[:, idxs]

    # merge primary & secondary edges
    edge_idxs = np.concatenate([edge_idxs_pri, edge_idxs_sec], axis=-1)

    return edge_idxs


@profile
def find_edges_by_nthres(cord_mat, nedg_thres):
    """Find edges by the <number_of_edges_per_node> threshold."""

    # initialization
    seq_len = cord_mat.shape[0]

    # primary edges (adjacent residues)
    edge_idxs_pri = np.stack([
        np.concatenate([np.arange(seq_len - 1), np.arange(1, seq_len)]),
        np.concatenate([np.arange(1, seq_len), np.arange(seq_len - 1)]),
    ], axis=0)

    # secondary edges (CA-CA distance lower than the threshold)
    mask_mat = (np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]) > 1)
    dist_mat = cdist(cord_mat, cord_mat, metric='euclidean')
    dist_mat_ext = dist_mat + (1 - mask_mat) * np.max(dist_mat)
    idxs_mat = np.argpartition(dist_mat_ext, nedg_thres, axis=-1)
    edge_idxs_sec = np.stack([
        np.repeat(np.arange(seq_len)[:, None], nedg_thres, axis=1).ravel(),
        idxs_mat[:, :nedg_thres].ravel(),
    ], axis=0)

    # merge primary & secondary edges
    edge_idxs = np.concatenate([edge_idxs_pri, edge_idxs_sec], axis=-1)

    return edge_idxs


@profile
def find_edges_by_sep(seq_len, sep):
    """Find edges by the residue separation in the amino-acid sequence."""

    # find edges by the residue seperation in the amino-acid sequence
    assert sep >= 1, 'residue separation must be positive; %d is given' % sep
    edge_idxs = np.stack([
        np.concatenate([np.arange(seq_len - sep), np.arange(sep, seq_len)]),
        np.concatenate([np.arange(sep, seq_len), np.arange(seq_len - sep)]),
    ], axis=0)

    return edge_idxs


@profile
def build_2d_inputs(core_data):
    """Build 2D inputs from the core data.

    Args:
    * core_data: dict of the core data

    Returns:
    * inputs: dict of 2D inputs
    """

    # initialization
    label = core_data['label']
    config = core_data['config']
    batch_size = config.batch_size
    seq_len = len(core_data['seq'])
    crop_size = seq_len if config.crop_mode == 'none' else config.crop_size

    # get residue indices' boundaries for random cropping
    crop_bnds = get_crop_bnds(seq_len, config.crop_mode, config.crop_size, batch_size)

    # concatenate pointwise & pairwise features to form conditional inputs
    cond_tns_raw_np = np.transpose(np.concatenate([
        np.repeat(core_data['fmpt'][None, :, :], seq_len, axis=0),
        np.repeat(core_data['fmpt'][:, None, :], seq_len, axis=1),
        core_data['ftpr'],
    ], axis=-1), [2, 0, 1])
    cond_tns_raw = torch.tensor(cond_tns_raw_np, dtype=torch.float32)
    n_dims_cond = cond_tns_raw.shape[0]

    # pack masks, 3D coordinates, and gradients into PyTorch tensors
    mask_vec = torch.tensor(core_data['mask'], dtype=torch.float32)
    cord_tns = torch.tensor(core_data['cord_p'], dtype=torch.float32)
    if 'grad' in core_data:
        grad_tns = torch.tensor(core_data['grad'], dtype=torch.float32)

    # build tensors for input features, 3D coordinates, and gradients
    cond_tns = torch.zeros((batch_size, n_dims_cond, crop_size, crop_size), dtype=torch.float32)
    mask_mat_pri = torch.zeros((batch_size, crop_size), dtype=torch.float32)
    mask_mat_sec = torch.zeros((batch_size, crop_size), dtype=torch.float32)
    cord_tns_pri = torch.zeros((batch_size, crop_size, 3), dtype=torch.float32)
    cord_tns_sec = torch.zeros((batch_size, crop_size, 3), dtype=torch.float32)
    grad_tns_pri = torch.zeros((batch_size, crop_size, 3), dtype=torch.float32)
    grad_tns_sec = torch.zeros((batch_size, crop_size, 3), dtype=torch.float32)
    for idx in range(batch_size):
        # obtain row-wise & column-wise boundaries
        ir_beg = crop_bnds[idx][0][0]
        ir_end = crop_bnds[idx][0][1]
        ic_beg = crop_bnds[idx][1][0]
        ic_end = crop_bnds[idx][1][1]

        # gather 3D coordinates & gradients
        cond_tns[idx] = cond_tns_raw[:, ir_beg:ir_end, ic_beg:ic_end]
        mask_mat_pri[idx] = mask_vec[ir_beg:ir_end]
        mask_mat_sec[idx] = mask_vec[ic_beg:ic_end]
        cord_tns_pri[idx] = cord_tns[idx, ir_beg:ir_end]
        cord_tns_sec[idx] = cord_tns[idx, ic_beg:ic_end]
        if 'grad' in core_data:
            grad_tns_pri[idx] = grad_tns[idx, ir_beg:ir_end]
            grad_tns_sec[idx] = grad_tns[idx, ic_beg:ic_end]
    dist_tns = torch.cdist(cord_tns_pri, cord_tns_sec)
    denc_tns = get_denc_tns_prob(dist_tns, config.n_dims_denc) \
        if config.denc_mthd == 'prob' else get_denc_tns_sigm(dist_tns, config.n_dims_denc)
    feat_tns = torch.cat([cond_tns, denc_tns], dim=1)

    # pack 2D inputs into a dict
    inputs = {}
    inputs['cond'] = cond_tns
    inputs['feat'] = feat_tns
    inputs['mask_p'] = mask_mat_pri
    inputs['mask_s'] = mask_mat_sec
    inputs['cord_p'] = cord_tns_pri
    inputs['cord_s'] = cord_tns_sec
    inputs['label'] = core_data['label']
    if 'grad' in core_data:
        inputs['grad_p'] = grad_tns_pri
        inputs['grad_s'] = grad_tns_sec
    if 'idxs' in core_data:
        inputs['idxs'] = torch.tensor(core_data['idxs'], dtype=torch.int64)
        inputs['stds'] = torch.tensor(core_data['stds'], dtype=torch.float32)

    return inputs


@profile
def update_2d_inputs(inputs, core_data, cord_tns):
    """Update 2D inputs with specified 3D coordinates.

    Args:
    * inputs: dict of 2D inputs
    * core_data: dict of the core data
    * cord_tns: 3D coordinates of size BS x L x 3

    Returns:
    * inputs: dict of updated 2D inputs
    """

    config = core_data['config']
    dist_tns = torch.cdist(cord_tns, cord_tns)
    denc_tns = get_denc_tns_prob(dist_tns, config.n_dims_denc) \
        if config.denc_mthd == 'prob' else get_denc_tns_sigm(dist_tns, config.n_dims_denc)
    inputs['feat'] = torch.cat([inputs['cond'].to(denc_tns.device), denc_tns], dim=1)
    inputs['cord_p'] = cord_tns

    return inputs


@profile
def build_3d_inputs(core_data, cord_tns=None):
    """Build 3D inputs from the core data, optionally with specified 3D coordinates.

    Args:
    * core_data: dict of the core data
    * cord_tns: (optional) 3D coordinates of size BS x L x 3

    Returns:
    * inputs: dict of 3D inputs
    """

    # initialization
    config = core_data['config']
    batch_size = config.batch_size
    n_nodes = len(core_data['seq'])

    # use <core_data['cord_p']> if <cord_tns> is not provided
    if cord_tns is None:
        cord_tns = core_data['cord_p']

    # build a DGL graph for each sample in the mini-batch
    graph_list = []
    for idx in range(batch_size):
        # find edges, and then determine their features & relative 3D coordinates
        cord_mat = cord_tns[idx]
        edge_idxs = find_edges_by_dthres(cord_mat, config.dist_thres, config.n_edges_max)
        #edge_idxs = find_edges_by_nthres(cord_mat, config.nedg_thres)
        edge_feats = core_data['ftpr'][edge_idxs[0], edge_idxs[1]]
        dcrd_mat = cord_mat[edge_idxs[0]] - cord_mat[edge_idxs[1]]

        # create a DGL graph
        graph = dgl.DGLGraph()
        graph.add_nodes(n_nodes)
        graph.add_edges(edge_idxs[0], edge_idxs[1])

        # add node/edge features & ground-truth labels to the graph
        graph.ndata['x'] = torch.tensor(cord_mat, dtype=torch.float32)
        graph.ndata['z'] = torch.tensor(core_data['cord_t'], dtype=torch.float32)
        graph.ndata['f'] = torch.tensor(core_data['fmpt'][..., None], dtype=torch.float32)
        graph.ndata['m'] = torch.tensor(core_data['mask'], dtype=torch.float32)
        if 'grad' in core_data:
            graph.ndata['y'] = torch.tensor(core_data['grad'][idx], dtype=torch.float32)
        graph.edata['d'] = torch.tensor(dcrd_mat, dtype=torch.float32)
        graph.edata['w'] = torch.tensor(edge_feats, dtype=torch.float32)

        # record the current graph
        graph_list.append(graph)

    # pack into a dict
    inputs = {}
    inputs['graph'] = dgl.batch(graph_list)
    if 'idxs' in core_data:
        idxs_ext = core_data['idxs'][:, None] * np.ones((1, n_nodes), dtype=np.int64)
        stds_ext = core_data['stds'][:, None] * np.ones((1, n_nodes), dtype=np.float32)
        inputs['idxs'] = torch.tensor(idxs_ext.ravel(), dtype=torch.int64)
        inputs['stds'] = torch.tensor(stds_ext.ravel(), dtype=torch.float32)

    return inputs


@profile
def build_3ds_inputs(core_data, cord_tns=None):
    """Build 3DS inputs from the core data, optionally with specified 3D coordinates.

    Args:
    * core_data: dict of the core data
    * cord_tns: (optional) 3D coordinates of size BS x L x 3

    Returns:
    * inputs: dict of 3D inputs

    Note:
    Multiple DGL graphs are constructed, one per residue separation.
    """

    # initialization
    config = core_data['config']
    batch_size = config.batch_size
    n_nodes = len(core_data['seq'])

    # use <core_data['cord_p']> if <cord_tns> is not provided
    if cord_tns is None:
        cord_tns = core_data['cord_p']

    # build a DGL graph for each sample in the mini-batch
    graph_dict = {}
    for sep in config.sep_list:
        # build a DGL graph for each sample in the mini-batch
        graph_list = []
        edge_idxs = find_edges_by_sep(n_nodes, sep)
        for idx in range(batch_size):
            # find edges, and then determine their features & relative 3D coordinates
            cord_mat = cord_tns[idx]
            edge_feats = core_data['ftpr'][edge_idxs[0], edge_idxs[1]]
            dcrd_mat = cord_mat[edge_idxs[0]] - cord_mat[edge_idxs[1]]

            # create a DGL graph
            graph = dgl.DGLGraph()
            graph.add_nodes(n_nodes)
            graph.add_edges(edge_idxs[0], edge_idxs[1])

            # add node/edge features & ground-truth labels to the graph
            graph.ndata['x'] = torch.tensor(cord_mat, dtype=torch.float32)
            graph.ndata['z'] = torch.tensor(core_data['cord_t'], dtype=torch.float32)
            graph.ndata['f'] = torch.tensor(core_data['fmpt'][..., None], dtype=torch.float32)
            graph.ndata['m'] = torch.tensor(core_data['mask'], dtype=torch.float32)
            if 'grad' in core_data:
                graph.ndata['y'] = torch.tensor(core_data['grad'][idx], dtype=torch.float32)
            graph.edata['d'] = torch.tensor(dcrd_mat, dtype=torch.float32)
            graph.edata['w'] = torch.tensor(edge_feats, dtype=torch.float32)

            # record the current graph
            graph_list.append(graph)

        # fuse multiple DGL graphs into one
        graph_dict['sep-%d' % sep] = dgl.batch(graph_list)

    # pack into a dict
    inputs = {}
    inputs['graph'] = graph_dict
    if 'idxs' in core_data:
        idxs_ext = core_data['idxs'][:, None] * np.ones((1, n_nodes), dtype=np.int64)
        stds_ext = core_data['stds'][:, None] * np.ones((1, n_nodes), dtype=np.float32)
        inputs['idxs'] = torch.tensor(idxs_ext.ravel(), dtype=torch.int64)
        inputs['stds'] = torch.tensor(stds_ext.ravel(), dtype=torch.float32)

    return inputs


class EbmDatasetConfig():
    """Configurations for the <EbmDataset> class."""

    def __init__(
            self,
            source=None,        # data source ('cath' / 'rcsb')
            input_frmt='2d',    # input format ('2d' / '3d')
            hdf_dpath=None,     # directory path to HDF5 files
            npz_dpath=None,     # directory path to NPZ files
            pid_fpath=None,     # path to protein IDs in the whitelist
            n_dims_onht=20,     # number of dimensions for one-hot encodings of amino-acids
            n_dims_penc=24,     # number of dimensions for positional encodings
            n_dims_dist=37,     # number of dimensions for inter-residue distance predictions
            n_dims_angl=64,     # number of dimensions for inter-residue orientation predictions
            exec_mode='train',  # execution mode ('train' / 'sample')
            batch_size=16,      # batch size
            noise_stds=None,    # random noise's standard deviations
            filt_mthd='none',   # filtering method ('none' / 'pcnt' / 'pcut')
            pcnt_vals=None,     # PCNT thresholds (ignored if <filt_mthd> is not 'pcnt')
            pcut_vals=None,     # PCUT thresholds (ignored if <filt_mthd> is not 'pcut')
            seq_len_min=-1,     # minimal FASTA sequence length (-1: unlimited)
            seq_len_max=-1,     # maximal FASTA sequence length (-1: unlimited)
            n_dims_denc=61,     # number of dimensions for distance encodings (2D only)
            denc_mthd='prob',   # distance encoding method (2D only)
            crop_mode='none',   # random cropping mode ('none' / 'single' / 'multi'; 2D only)
            crop_size=-1,       # random cropping size (2D only)
            dist_thres=8.0,     # distance threshold for building edges (3D only)
            n_edges_max=2000,   # maximal number of edges in a single DGL graph (3D only)
            nedg_thres=8,       # maximal number of edges per node for building edges (3D only)
            sep_list=None,      # list of residual separation for building edges (3D only)
        ):
        """Constructor function."""

        # setup configurations
        self.source = source
        self.input_frmt = input_frmt
        self.hdf_dpath = hdf_dpath
        self.npz_dpath = npz_dpath
        self.pid_fpath = pid_fpath
        self.n_dims_onht = n_dims_onht
        self.n_dims_penc = n_dims_penc
        self.n_dims_dist = n_dims_dist
        self.n_dims_angl = n_dims_angl
        self.exec_mode = exec_mode
        self.batch_size = batch_size
        self.noise_stds = noise_stds if noise_stds is not None else []
        self.filt_mthd = filt_mthd
        self.pcnt_vals = pcnt_vals if pcnt_vals is not None else []
        self.pcut_vals = pcut_vals if pcut_vals is not None else []
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max
        self.n_dims_denc = n_dims_denc
        self.denc_mthd = denc_mthd
        self.crop_mode = crop_mode
        self.crop_size = crop_size
        self.dist_thres = dist_thres
        self.n_edges_max = n_edges_max
        self.nedg_thres = nedg_thres
        self.sep_list = sep_list if sep_list is not None else []

        # over-ride the minimal FASTA sequence length if random cropping is enabled
        if self.crop_mode != 'none':
            self.seq_len_min = max(self.seq_len_min, self.crop_size)

        # validate configurations
        assert self.source in ['cath', 'rcsb'], 'unrecognized data source: ' + self.source
        assert self.input_frmt in ['2d', '3d', '3ds'], \
            'unrecognized input format: ' + self.input_frmt
        assert self.exec_mode in ['train', 'sample', 'finetune'], \
            'unrecognized execution mode: ' + self.exec_mode
        assert self.filt_mthd in ['none', 'pcut', 'pcnt'], \
            'unrecognized filtering method: ' + self.filt_mthd
        assert self.denc_mthd in ['prob', 'sigm'], \
            'unrecognized distance encoding method: ' + self.denc_mthd
        assert self.crop_mode in ['none', 'single', 'multi'], \
            'unrecognized random cropping mode: ' + self.crop_mode


    def show(self):
        """Show detailed configurations."""

        logging.info('=== EbmDatasetConfig - Start ===')
        logging.info('source: %s', self.source)
        logging.info('input_frmt: %s', self.input_frmt)
        logging.info('hdf_dpath: %s', self.hdf_dpath)
        logging.info('npz_dpath: %s', self.npz_dpath)
        logging.info('n_dims_onht: %d', self.n_dims_onht)
        logging.info('n_dims_penc: %d', self.n_dims_penc)
        logging.info('n_dims_dist: %d', self.n_dims_dist)
        logging.info('n_dims_angl: %d', self.n_dims_angl)
        logging.info('exec_mode: %s', self.exec_mode)
        logging.info('batch_size: %d', self.batch_size)
        logging.info('noise_stds: %s', ','.join(['%.2e' % x for x in self.noise_stds]))
        logging.info('filt_mthd: %s', self.filt_mthd)
        logging.info('pcnt_vals: %s', ','.join(['%.2f' % x for x in self.pcnt_vals]))
        logging.info('pcut_vals: %s', ','.join(['%.2f' % x for x in self.pcut_vals]))
        logging.info('seq_len_min: %d', self.seq_len_min)
        logging.info('seq_len_max: %d', self.seq_len_max)
        logging.info('n_dims_denc: %d', self.n_dims_denc)
        logging.info('denc_mthd: %s', self.denc_mthd)
        logging.info('crop_mode: %s', self.crop_mode)
        logging.info('crop_size: %d', self.crop_size)
        logging.info('dist_thres: %.2f', self.dist_thres)
        logging.info('n_edges_max: %d', self.n_edges_max)
        logging.info('nedg_thres: %d', self.nedg_thres)
        logging.info('sep_list: %s', ','.join(['%d' % x for x in self.sep_list]))
        logging.info('=== EbmDatasetConfig - Finish ===')


class EbmDataset(Dataset):
    """Dataset for EBM training & sampling - 2D & 3D inputs."""

    def __init__(self, config):
        """Constructor function."""

        # setup configurations
        self.config = config
        self.config.show()

        # create the CATH-PDB / RCSB-PDB dataset
        if self.config.source == 'cath':
            self.dataset = self.__create_cath_dataset()
        elif self.config.source == 'rcsb':
            self.dataset = self.__create_rcsb_dataset()
        else:
            raise ValueError('unrecognized data source: %s' % self.config.source)
        logging.info('# of elements in the dataset: %d', len(self.dataset))

        # setup the maximal sequence length for positional encodings
        self.penc_seq_len_max = 1000


    def __len__(self):
        """Get the number of elements in the dataset."""

        return len(self.dataset)


    @profile
    def __getitem__(self, idx):
        """Get the i-th element in the dataset."""

        # build a data dict from the core data
        core_data = self.dataset[idx]
        data_dict = self.__build_data_dict(core_data)

        # build 2D/3D inputs from the data dict
        if self.config.input_frmt == '2d':
            inputs = build_2d_inputs(data_dict)
        elif self.config.input_frmt == '3d':
            inputs = build_3d_inputs(data_dict)
        else:  # then <self.config.input_frmt> must be '3ds'
            inputs = build_3ds_inputs(data_dict)

        return inputs, data_dict


    @profile
    def __create_cath_dataset(self):
        """Create the CATH-PDB dataset."""

        config = CathPdbDatasetConfig(
            self.config.hdf_dpath,
            self.config.npz_dpath,
            did_fpath=self.config.pid_fpath,
            use_dist=(self.config.n_dims_dist > 0),
            use_angl=(self.config.n_dims_angl > 0),
            seq_len_min=self.config.seq_len_min,
            seq_len_max=self.config.seq_len_max,
        )

        return CathPdbDataset(config)


    @profile
    def __create_rcsb_dataset(self):
        """Create the RCSB-PDB dataset."""

        config = RcsbPdbDatasetConfig(
            self.config.hdf_dpath,
            self.config.npz_dpath,
            cid_fpath=self.config.pid_fpath,
            use_dist=(self.config.n_dims_dist > 0),
            use_angl=(self.config.n_dims_angl > 0),
            seq_len_min=self.config.seq_len_min,
            seq_len_max=self.config.seq_len_max,
        )

        return RcsbPdbDataset(config)


    @profile
    def __build_data_dict(self, core_data):
        """Build a data dict from the core data (id + seq + cord + mask + dist + angl)."""

        # build pointwise & pairwise features
        seq_len = len(core_data['seq'])
        onht_mat = self.__get_onht_mat(core_data['seq'])
        penc_mat = self.__get_penc_mat(seq_len)
        feat_mat_pont = np.concatenate([onht_mat, penc_mat], axis=-1)
        feat_tns_pair = np.concatenate([core_data['dist'], core_data['angl']], axis=-1)

        # build 3D coordinates (random perturbation OR random initialization)
        if self.config.exec_mode == 'train':
            data_dict_addi = self.__get_data_dict_addi_pert(core_data['cord'], core_data['mask'])
        elif self.config.exec_mode == 'finetune':
            data_dict_addi = self.__get_data_dict_addi_finetune(core_data['cord'], core_data['mask'])
        else:  # then <self.config.exec_mode> must be 'sample'
            data_dict_addi = self.__get_data_dict_addi_rand(core_data['cord'], core_data['mask'])

        # pack all the essential data into a dict
        data_dict_finl = {
            'id': core_data['id'],
            'seq': core_data['seq'],
            'fmpt': feat_mat_pont,
            'ftpr': feat_tns_pair,
            'config': self.config,
            'label': core_data['label'],
            **data_dict_addi,
        }

        return data_dict_finl


    @profile
    def __get_onht_mat(self, aa_seq):
        """Get one-hot encodings of amino-acids."""

        seq_len = len(aa_seq)
        if self.config.n_dims_onht == 0:
            onht_mat = np.zeros((seq_len, 0), dtype=np.float32)
        else:
            # aa_seq1 = str(aa_seq, encoding = "utf-8") # gai
            # aa_idxs = np.array([AA_NAMES_1CHAR.index(x) for x in aa_seq1], dtype=np.int32)
            aa_idxs = np.array([AA_NAMES_1CHAR.index(x) for x in aa_seq], dtype=np.int32)
            onht_mat = cvt_to_one_hot(aa_idxs, len(AA_NAMES_1CHAR))  # L x 20

        return onht_mat


    @profile
    def __get_penc_mat(self, seq_len):
        """Get positional encodings (NeurIPS's original design)."""

        if self.config.n_dims_penc == 0:
            penc_mat = np.zeros((seq_len, 0), dtype=np.float32)
        else:
            assert self.config.n_dims_penc % 2 == 0 and self.config.n_dims_penc // 2 >= 2
            n_levls = self.config.n_dims_penc // 2
            div_fctrs = np.power(self.penc_seq_len_max, np.arange(n_levls) / (n_levls - 1))
            penc_mat = np.concatenate([
                np.sin(np.arange(seq_len)[:, None] / div_fctrs[None, :]),
                np.cos(np.arange(seq_len)[:, None] / div_fctrs[None, :]),
            ], axis=-1)  # L x D

        return penc_mat


    @profile
    def __get_data_dict_addi_pert(self, atom_cords, atom_masks):
        """Get an additional data dict of perturbed 3D coordinates."""

        # initialization
        seq_len = atom_cords.shape[0]
        mask_vec = atom_masks[:, 1]
        cord_mat_true = atom_cords[:, 1]  # N-CA-C

        # randomly perturb 3D coordinates and compute corresponding gradients
        idxs_levl = np.random.randint(self.config.noise_stds.size, size=self.config.batch_size)
        noise_stds = self.config.noise_stds[idxs_levl]
        cord_tns_pert = np.zeros((self.config.batch_size, seq_len, 3), dtype=np.float32)
        grad_tns = np.zeros((self.config.batch_size, seq_len, 3), dtype=np.float32)
        for idx in range(self.config.batch_size):
            cord_mat_pert = cord_mat_true \
                + np.random.normal(scale=noise_stds[idx], size=cord_mat_true.shape)
            cord_tns_pert[idx] = cord_mat_pert
            #cord_tns_pert[idx] = self.__project_3d_cords(cord_mat_true, cord_mat_pert)
            grad_tns[idx] = (cord_mat_true - cord_tns_pert[idx]) / noise_stds[idx] ** 2

        # pack into a dict
        data_dict = {
            'mask': mask_vec,
            'cord_t': cord_mat_true,
            'cord_p': cord_tns_pert,
            'grad': grad_tns,
            'idxs': idxs_levl,
            'stds': noise_stds,
        }

        return data_dict


    @profile
    def __get_data_dict_addi_rand(self, atom_cords, atom_masks):
        """Get an additional data dict of randomized 3D coordinates."""

        # initialization
        seq_len = atom_cords.shape[0]
        mask_vec = atom_masks[:, 1]
        cord_mat_true = atom_cords[:, 1]  # N-CA-C

        # randomly initialize 3D coordinates
        cord_tns_pert = np.random.normal(
            size=(self.config.batch_size, seq_len, 3)).astype(np.float32)

        # pack into a dict
        data_dict = {
            'mask': mask_vec,
            'cord_t': cord_mat_true,
            'cord_p': cord_tns_pert,
        }

        return data_dict


    @profile
    def __get_data_dict_addi_finetune(self, atom_cords, atom_masks):
        """Get an additional data dict of perturbed 3D coordinates."""

        # initialization
        seq_len = atom_cords.shape[0]
        mask_vec = atom_masks[:, 1]
        cord_mat_true = atom_cords[:, 1]  # N-CA-C

        # randomly perturb 3D coordinates and compute corresponding gradients
        idxs_levl = np.random.randint(self.config.noise_stds.size, size=self.config.batch_size)
        noise_stds = self.config.noise_stds[idxs_levl]
        cord_tns_pert = np.zeros((self.config.batch_size, seq_len, 3), dtype=np.float32)
        grad_tns = np.zeros((self.config.batch_size, seq_len, 3), dtype=np.float32)
        for idx in range(self.config.batch_size):
            # cord_mat_pert = cord_mat_true \
            #     + np.random.normal(scale=noise_stds[idx], size=cord_mat_true.shape)

            cord_mat_pert = cord_mat_true
            cord_tns_pert[idx] = cord_mat_pert

            #cord_tns_pert[idx] = self.__project_3d_cords(cord_mat_true, cord_mat_pert)
            grad_tns[idx] = (cord_mat_true - cord_tns_pert[idx]) / noise_stds[idx] ** 2

        # pack into a dict
        data_dict = {
            'mask': mask_vec,
            'cord_t': cord_mat_true,
            'cord_p': cord_tns_pert,
            'grad': grad_tns,
            'idxs': idxs_levl,
            'stds': noise_stds,
        }

        return data_dict

    @classmethod
    def __project_3d_cords(cls, cord_mat_true, cord_mat_pert):
        """Project 3D coordinates so that adjacent CA-CA distance is preserved."""

        seq_len = cord_mat_true.shape[0]
        dist_vec_true = np.linalg.norm(cord_mat_true[:-1] - cord_mat_true[1:], axis=-1)
        dist_vec_pert = np.linalg.norm(cord_mat_pert[:-1] - cord_mat_pert[1:], axis=-1)
        cord_mat_proj = np.copy(cord_mat_pert)
        for idx in range(seq_len - 1, 0, -1):  # no need to adjust the first atom
            alpha = dist_vec_true[idx - 1] / dist_vec_pert[idx - 1]
            cord_mat_proj[idx:] += (alpha - 1.0) * (cord_mat_proj[idx] - cord_mat_proj[idx - 1])

        return cord_mat_proj
