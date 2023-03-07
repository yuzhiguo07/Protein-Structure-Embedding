"""Unit-tests for <RcsbPdbDataset>."""

import os
import logging

from tfold_se3.utils import tfold_init
from tfold_se3.datasets.rcsb_pdb.rcsb_pdb_dataset import RcsbPdbDatasetConfig
from tfold_se3.datasets.rcsb_pdb.rcsb_pdb_dataset import RcsbPdbDataset


def test_rcsb_pdb_dataset(config):
    """Run unit-tests for <RcsbPdbDataset>."""

    dataset = RcsbPdbDataset(config)
    n_prots = len(dataset)
    logging.info('RcsbPdbDataset - # of elements: %d', n_prots)
    for idx_prot in range(n_prots):
        data_dict = dataset[idx_prot]
        logging.info('id: %s', data_dict['id'])
        logging.info('seq: %s', data_dict['seq'])
        logging.info('cord: %s / %s', str(data_dict['cord'].shape), str(data_dict['cord'].dtype))
        logging.info('mask: %s / %s', str(data_dict['mask'].shape), str(data_dict['mask'].dtype))
        logging.info('dist: %s / %s', str(data_dict['dist'].shape), str(data_dict['dist'].dtype))
        logging.info('angl: %s / %s', str(data_dict['angl'].shape), str(data_dict['angl'].dtype))


def main():
    """Main entry."""

    # load configurations from the YAML file
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_fpath = os.path.join(curr_dir, 'config_gpu241.yaml')
    config = tfold_init(cfg_fpath)

    # construct configurations for <RcsbPdbDataset> & <RcsbPdbIterableDataset>
    ds_config = RcsbPdbDatasetConfig(
        hdf_dpath=config['hdf_dpath'],
        npz_dpath=config['npz_dpath'],
        cid_fpath=config['cid_fpath_tst'],
        seq_len_min=-1,
        seq_len_max=-1,
    )

    # run unit-tests for <RcsbPdbDataset>
    test_rcsb_pdb_dataset(ds_config)


if __name__ == '__main__':
    main()
