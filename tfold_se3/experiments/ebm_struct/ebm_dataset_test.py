"""Unit-tests for the <EbmDataset> class."""

import os
import logging

import torch
import numpy as np

from tfold_se3.utils import tfold_init
from tfold_se3.experiments.ebm_struct.ebm_dataset import EbmDatasetConfig
from tfold_se3.experiments.ebm_struct.ebm_dataset import EbmDataset
from tfold_se3.experiments.ebm_struct.ebm_dataset import update_2d_inputs
from tfold_se3.experiments.ebm_struct.ebm_dataset import build_3d_inputs
from tfold_se3.experiments.ebm_struct.ebm_dataset import build_3ds_inputs
from tfold_se3.experiments.ebm_struct.utils import calc_noise_stds


def get_ds_config_base(config):
    """Get <EbmDataset>'s configurations for 2D/3D/3DS inputs."""

    ds_config = {
        'source': config['data_source'],
        'hdf_dpath': config['hdf_dpath_cath'],
        'npz_dpath': config['npz_dpath_cath'],
        'n_dims_onht': config['n_dims_onht'] if config['use_onht'] else 0,
        'n_dims_penc': config['n_dims_penc'] if config['use_penc'] else 0,
        'n_dims_dist': config['n_dims_dist'] if config['use_dist'] else 0,
        'n_dims_angl': config['n_dims_angl'] if config['use_angl'] else 0,
        'filt_mthd': config['filt_mthd'],
        'pcnt_vals': config['pcnt_vals'],
        'pcut_vals': config['pcut_vals'],
        'seq_len_min': config['seq_len_min'],
        'seq_len_max': config['seq_len_max'],
    }

    return ds_config


def get_ds_config_2d(config):
    """Get <EbmDataset>'s configurations for 2D inputs."""

    ds_config_base = get_ds_config_base(config)
    ds_config = {
        'input_frmt': '2d',
        'n_dims_denc': config['n_dims_denc'],
        'denc_mthd': config['denc_mthd'],
        **ds_config_base,
    }

    return ds_config


def get_ds_config_3d(config):
    """Get <EbmDataset>'s configurations for 3D inputs."""

    ds_config_base = get_ds_config_base(config)
    ds_config = {
        'input_frmt': '3d',
        'dist_thres': config['dist_thres'],
        'n_edges_max': config['n_edges_max'],
        **ds_config_base,
    }

    return ds_config


def get_ds_config_3ds(config):
    """Get <EbmDataset>'s configurations for 3D inputs."""

    ds_config_base = get_ds_config_base(config)
    ds_config = {
        'input_frmt': '3ds',
        'sep_list': config['sep_list'],
        **ds_config_base,
    }

    return ds_config


def get_ds_config_trn(config, ds_config_base, noise_stds):
    """Get <EbmDataset>'s configurations for the training subset."""

    ds_config = EbmDatasetConfig(
        pid_fpath=config['pid_fpath_cath_trn'],
        exec_mode='train',
        batch_size=config['batch_size_trn'],
        noise_stds=noise_stds,
        crop_mode=config['crop_mode'],
        crop_size=config['crop_size'],
        **ds_config_base,
    )

    return ds_config


def get_ds_config_val(config, ds_config_base, noise_stds):
    """Get <EbmDataset>'s configurations for the training subset."""

    ds_config = EbmDatasetConfig(
        pid_fpath=config['pid_fpath_cath_val'],
        exec_mode='train',
        batch_size=config['batch_size_val'],
        noise_stds=noise_stds,
        crop_mode=config['crop_mode'],
        crop_size=config['crop_size'],
        **ds_config_base,
    )

    return ds_config


def get_ds_config_tst(config, ds_config_base):
    """Get <EbmDataset>'s configurations for the training subset."""

    ds_config = EbmDatasetConfig(
        pid_fpath=config['pid_fpath_cath_tst'],
        exec_mode='sample',
        batch_size=config['batch_size_tst'],
        **ds_config_base,
    )

    return ds_config


def inspect_data_dict(data_dict):
    """Inspect the data dict."""

    for key, val in data_dict.items():
        if not isinstance(val, (torch.Tensor, np.ndarray)):
            logging.info('%s: %s', key, val)
        else:
            logging.info('%s: %s / %s', key, val.shape, val.dtype)


def test_ebm_dataset_2d(config, n_smpls=1):
    """Test <EbmDataset> for 2D inputs with specified configurations."""

    dataset = EbmDataset(config)
    logging.info('# of elements in the dataset: %d', len(dataset))
    for idx in range(min(n_smpls, len(dataset))):
        # obtain 2D inputs and core data dict
        inputs, core_data = dataset[idx]

        # inspect 2D inputs
        logging.info('=== 2D inputs ===')
        inspect_data_dict(inputs)

        # inspect the data dict of core information for building 2D/3D/3DS inputs
        logging.info('=== core data dict ===')
        inspect_data_dict(core_data)


def test_ebm_dataset_3d(config, n_smpls=1):
    """Test <EbmDataset> for 3D inputs with specified configurations."""

    dataset = EbmDataset(config)
    logging.info('# of elements in the dataset: %d', len(dataset))
    for idx in range(min(n_smpls, len(dataset))):
        # obtain 3D inputs and core data dict
        inputs, core_data = dataset[idx]

        # inspect 3D inputs
        logging.info('=== 3D inputs ===')
        inspect_data_dict(inputs)

        # inspect the data dict of core information for building 2D/3D/3DS inputs
        logging.info('=== core data dict ===')
        inspect_data_dict(core_data)


def test_ebm_dataset_3ds(config, n_smpls=1):
    """Test <EbmDataset> for 3DS inputs with specified configurations."""

    dataset = EbmDataset(config)
    logging.info('# of elements in the dataset: %d', len(dataset))
    for idx in range(min(n_smpls, len(dataset))):
        # obtain 3DS inputs and core data dict
        inputs, core_data = dataset[idx]

        # inspect 3DS inputs
        logging.info('=== 3DS inputs ===')
        inspect_data_dict(inputs)

        # inspect the data dict of core information for building 2D/3D/3DS inputs
        logging.info('=== core data dict ===')
        inspect_data_dict(core_data)


def test_update_2d_inputs(config):
    """Test <update_2d_inputs> with specified configurations."""

    dataset = EbmDataset(config)
    inputs_old, core_data = dataset[np.random.randint(len(dataset))]
    cord_tns_np = np.random.normal(size=core_data['cord_p'].shape)
    cord_tns = torch.tensor(cord_tns_np, dtype=torch.float32)
    inputs_new = update_2d_inputs(inputs_old, core_data, cord_tns)
    logging.info('=== 2D inputs (old) ===')
    inspect_data_dict(inputs_old)
    logging.info('=== 2D inputs (new) ===')
    inspect_data_dict(inputs_new)


def test_build_3d_inputs(config):
    """Test <build_3d_inputs> with specified configurations."""

    dataset = EbmDataset(config)
    inputs_old, core_data = dataset[np.random.randint(len(dataset))]
    graph = inputs_old['graph']
    cord_mat = np.random.normal(size=graph.ndata['x'].shape).astype(np.float32)
    cord_tns = np.reshape(cord_mat, [config.batch_size, -1, 3])
    inputs_new = build_3d_inputs(core_data, cord_tns)
    logging.info('=== 3D inputs (old) ===')
    inspect_data_dict(inputs_old)
    logging.info('=== 3D inputs (new) ===')
    inspect_data_dict(inputs_new)


def test_build_3ds_inputs(config):
    """Test <build_3ds_inputs> with specified configurations."""

    dataset = EbmDataset(config)
    inputs_old, core_data = dataset[np.random.randint(len(dataset))]
    graph = inputs_old['graph']['sep-1']
    cord_mat = np.random.normal(size=graph.ndata['x'].shape).astype(np.float32)
    cord_tns = np.reshape(cord_mat, [config.batch_size, -1, 3])
    inputs_new = build_3ds_inputs(core_data, cord_tns)
    logging.info('=== 3DS inputs (old) ===')
    inspect_data_dict(inputs_old)
    logging.info('=== 3DS inputs (new) ===')
    inspect_data_dict(inputs_new)


def main():
    """Main entry."""

    # initialize the tFold-SE3 framework
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_fpath = os.path.join(curr_dir, 'config.yaml')
    config = tfold_init(cfg_fpath)

    # setup the random noise's standard deviations
    noise_stds = calc_noise_stds(
        config['noise_std_max'], config['noise_std_min'], config['n_noise_levls'])

    # get <EbmDataset>'s configurations for 2D/3D inputs
    ds_config_2d = get_ds_config_2d(config)
    ds_config_3d = get_ds_config_3d(config)
    ds_config_3ds = get_ds_config_3ds(config)

    # test <EbmDataset> on the training subset
    logging.info('testing <EbmDataset> on the training subset (2D inputs) ...')
    ds_config = get_ds_config_trn(config, ds_config_2d, noise_stds)
    test_ebm_dataset_2d(ds_config, n_smpls=4)
    logging.info('testing <EbmDataset> on the training subset (3D inputs) ...')
    ds_config = get_ds_config_trn(config, ds_config_3d, noise_stds)
    test_ebm_dataset_3d(ds_config, n_smpls=4)
    logging.info('testing <EbmDataset> on the training subset (3DS inputs) ...')
    ds_config = get_ds_config_trn(config, ds_config_3ds, noise_stds)
    test_ebm_dataset_3ds(ds_config, n_smpls=4)

    # test <EbmDataset> on the validation subset
    logging.info('testing <EbmDataset> on the validation subset (2D inputs) ...')
    ds_config = get_ds_config_val(config, ds_config_2d, noise_stds)
    test_ebm_dataset_2d(ds_config, n_smpls=4)
    logging.info('testing <EbmDataset> on the validation subset (3D inputs) ...')
    ds_config = get_ds_config_val(config, ds_config_3d, noise_stds)
    test_ebm_dataset_3d(ds_config, n_smpls=4)
    logging.info('testing <EbmDataset> on the validation subset (3DS inputs) ...')
    ds_config = get_ds_config_val(config, ds_config_3ds, noise_stds)
    test_ebm_dataset_3ds(ds_config, n_smpls=4)

    # test <EbmDataset> on the test subset
    logging.info('testing <EbmDataset> on the test subset (2D inputs) ...')
    ds_config = get_ds_config_tst(config, ds_config_2d)
    test_ebm_dataset_2d(ds_config, n_smpls=4)
    test_update_2d_inputs(ds_config)
    logging.info('testing <EbmDataset> on the test subset (3D inputs) ...')
    ds_config = get_ds_config_tst(config, ds_config_3d)
    test_ebm_dataset_3d(ds_config, n_smpls=4)
    test_build_3d_inputs(ds_config)
    logging.info('testing <EbmDataset> on the test subset (3DS inputs) ...')
    ds_config = get_ds_config_tst(config, ds_config_3ds)
    test_ebm_dataset_3ds(ds_config, n_smpls=4)
    test_build_3ds_inputs(ds_config)


if __name__ == '__main__':
    main()
