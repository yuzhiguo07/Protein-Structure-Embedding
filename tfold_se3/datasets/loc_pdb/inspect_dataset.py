"""Inspect the dataset (stored as HDF5 files)."""

import os
import logging
from collections import defaultdict

import h5py
import numpy as np

from tfold_se3.utils import tfold_init


def inspect_dataset(hdf_fpath, log_fpath, verbose=False):
    """Inspect the dataset."""

    # inspect the HDF5 file for FASTA sequence & atom coordinates
    with h5py.File(hdf_fpath, 'r') as i_file:
        n_chains = len(i_file)
        if verbose:
            for chain_name in i_file.keys():
                logging.info('chain name: %s', chain_name)
                for key, dataset in i_file[chain_name].items():
                    val = dataset[()]
                    if key == 'aa_seq':
                        logging.info('- %s: %s / %d residues', key, val, len(val))
                    else:
                        logging.info('- %s: %s / %s', key, str(val.shape), str(val.dtype))
    logging.info('# of PDB-chain entries: %d (%s)', n_chains, os.path.basename(hdf_fpath))

    # inspect the LOG file for PDB parsing statistics
    status_dict = defaultdict(int)
    with open(log_fpath, 'r') as i_file:
        for i_line in i_file:
            status = i_line.split()[-1]
            status_dict[status] += 1

    return n_chains, status_dict

def main():
    """Main entry."""

    # load configurations from the YAML file
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_fpath = os.path.join(curr_dir, 'config.yaml')
    config = tfold_init(cfg_fpath)

    # inspect all the HDF5 files
    n_chains_list = []
    status_dict = defaultdict(int)
    hdf_fnames = sorted([x for x in os.listdir(config['hdf_dpath']) if x.endswith('.hdf5')])
    for hdf_fname in hdf_fnames:
        hdf_fpath = os.path.join(config['hdf_dpath'], hdf_fname)
        log_fpath = os.path.join(config['log_dpath'], hdf_fname.replace('.hdf5', '.log'))
        n_chains, status_dict_new = inspect_dataset(hdf_fpath, log_fpath, verbose=False)
        n_chains_list.append(n_chains)
        for status, count in status_dict_new.items():
            status_dict[status] += count
    logging.info('# of PDB-chain entries (in total): %d', sum(n_chains_list))
    logging.info('# of PDB-chain entries (per file): %.2f', np.mean(np.array(n_chains_list)))
    for status, count in status_dict.items():
        logging.info('%s: %d', status, count)

if __name__ == '__main__':
    main()
