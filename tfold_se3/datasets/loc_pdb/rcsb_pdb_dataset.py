"""RCSB-PDB dataset."""

import os
import random
import logging

import h5py
import numpy as np
from torch.utils.data import Dataset


class RcsbPdbDatasetConfig():
    """Configurations for <RcsbPdbDataset>."""

    def __init__(
            self,
            hdf_dpath,
            npz_dpath,
            cid_fpath=None,
            use_dist=True,
            use_angl=True,
            seq_len_min=-1,
            seq_len_max=-1,
        ):
        """Constructor function."""

        self.hdf_dpath = hdf_dpath
        self.npz_dpath = npz_dpath
        self.cid_fpath = cid_fpath
        self.use_dist = use_dist
        self.use_angl = use_angl
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max

    def show(self):
        """Show detailed configurations."""

        logging.info('=== RcsbPdbDatasetConfig - Start ===')
        logging.info('hdf_dpath: %s', self.hdf_dpath)
        logging.info('npz_dpath: %s', self.npz_dpath)
        logging.info('cid_fpath: %s', self.cid_fpath)
        logging.info('use_dist: %s', self.use_dist)
        logging.info('use_angl: %s', self.use_angl)
        logging.info('seq_len_min: %d', self.seq_len_min)
        logging.info('seq_len_max: %d', self.seq_len_max)
        logging.info('=== RcsbPdbDatasetConfig - Finish ===')


class RcsbPdbDataset(Dataset):
    """RCSB-PDB dataset (map-style)."""

    def __init__(self, config):
        """Constructor function."""

        # setup configurations
        self.config = config
        self.config.show()

        # build the mapping from element's index to (HDF5 file path, protein ID)
        self.i2f_map = self.__build_i2f_map(config)

    def __len__(self):
        """Get the total number of elements."""

        return len(self.i2f_map)

    def __getitem__(self, idx):
        """Get the i-th element."""

        # obtain HDF5 & NPZ file paths & protein ID
        hdf_fpath, prot_id = self.i2f_map[idx]


        # parse the HDF5 file
        with h5py.File(hdf_fpath, 'r', driver='core') as i_file:
            aa_seq = i_file[prot_id]['aa_seq'][()]
            atom_cords = i_file[prot_id]['atom_cords'][()]
            atom_masks = i_file[prot_id]['atom_masks'][()]

        # parse the NPZ file
        seq_len = len(aa_seq)
        pred_tns_null = np.zeros((seq_len, seq_len, 0), dtype=np.float32)
        if not self.config.use_dist and not self.config.use_angl:
            dist_tns = pred_tns_null
            angl_tns = pred_tns_null
        else:
            npz_dpath = os.path.join(self.config.npz_dpath, prot_id)
            npz_fnames = [x for x in os.listdir(npz_dpath) if x.endswith('.npz')]
            npz_fpath = os.path.join(npz_dpath, random.choice(npz_fnames))
            
            
            with np.load(npz_fpath) as npz_data:
                dist_tns = pred_tns_null if not self.config.use_dist else npz_data['dist']
                angl_tns = pred_tns_null if not self.config.use_angl else \
                    np.concatenate([npz_data['omega'], npz_data['theta'], npz_data['phi']], axis=-1)

        # pack all the essential data into dict
        data_dict = {
            'id': prot_id,
            'seq': aa_seq,
            'cord': atom_cords,
            'mask': atom_masks,
            'dist': dist_tns,
            'angl': angl_tns,
        }

        return data_dict

    @classmethod
    def __build_i2f_map(cls, config):
        """Build the mapping from element's index to the (HDF5 file path, protein ID) tuple."""

        # obtain a list of whitelist for protein IDs
        prot_ids = None
        if config.cid_fpath is not None:
            with open(config.cid_fpath, 'r') as i_file:
                prot_ids = [i_line.strip() for i_line in i_file]

        # build the mapping from element's index to the (HDF5 file path, protein ID) tuple
        n_prots = 0
        i2f_map = {}
        for hdf_fname in os.listdir(config.hdf_dpath):
            if not hdf_fname.endswith('.hdf5'):
                continue
            hdf_fpath = os.path.join(config.hdf_dpath, hdf_fname)
            with h5py.File(hdf_fpath, 'r', driver='core') as i_file:
                for prot_id in i_file:
                    if prot_ids is not None and prot_id not in prot_ids:
                        continue  # not in the whitelist
                    aa_seq = i_file[prot_id]['aa_seq'][()]
                    if config.seq_len_min != -1 and len(aa_seq) < config.seq_len_min:
                        continue  # shorter than the minimal sequence length
                    if config.seq_len_max != -1 and len(aa_seq) > config.seq_len_max:
                        continue  # longer than the maximal sequence length
                    i2f_map[n_prots] = (hdf_fpath, prot_id)
                    n_prots += 1

        return i2f_map
