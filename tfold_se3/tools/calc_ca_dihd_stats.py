"""Calculate CA atoms' dihedral angle statistics (for resolving the handedness issue)."""

import os
import re

import h5py
import numpy as np

from tfold_se3.utils import calc_dihedral_angle


def calc_dihd_angls_impl(atom_cords, atom_masks):
    """Calculate CA atoms' dihedral angles - core implementation."""

    dihd_angls = []
    seq_len = atom_cords.shape[0]
    ca_cords = atom_cords[:, 1]  # N - CA - C'
    for idx in range(seq_len - 4):
        if atom_masks[idx:idx + 4, 1].min() != 1:
            continue
        dihd_angls.append(calc_dihedral_angle(
            ca_cords[idx], ca_cords[idx + 1], ca_cords[idx + 2], ca_cords[idx + 3]))

    return dihd_angls

def calc_dihd_angls(path):
    """Calculate CA atoms' dihedral angles from the HDF5 file."""

    print('parsing the HDF5 file: ' + path)
    dihd_angls = []
    with h5py.File(path, 'r', driver='core') as i_file:
        for prot_id in i_file:
            atom_cords = i_file[prot_id]['atom_cords'][()]
            atom_masks = i_file[prot_id]['atom_masks'][()]
            dihd_angls_new = calc_dihd_angls_impl(atom_cords, atom_masks)
            dihd_angls.extend(dihd_angls_new)

    return dihd_angls


def main():
    """Main entry."""

    # configurations
    n_bins = 64
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    hdf_dpath = '/apdcephfs/private_jonathanwu_cq/Datasets/CATH-PDB/hdf5.files'
    stt_fpath = os.path.join(curr_dir, 'ca_dihd_stats.txt')

    # only use HDF5 files belonging to the training subsets
    regex = re.compile(r'^train-[0-9]+-of-[0-9]+\.hdf5$')
    hdf_fnames = [x for x in os.listdir(hdf_dpath) if re.search(regex, x)]
    hdf_fpaths = [os.path.join(hdf_dpath, x) for x in hdf_fnames]

    # calculate CA atoms' dihedral angles
    dihd_angls_raw = []
    for hdf_fpath in hdf_fpaths:
        dihd_angls_new = calc_dihd_angls(hdf_fpath)
        dihd_angls_raw.extend(dihd_angls_new)
    dihd_angls = np.array(dihd_angls_raw, dtype=np.float32)
    print('dihd. angle stats.: %.4f (min) / %.4f (max)' % (np.min(dihd_angls), np.max(dihd_angls)))

    # group CA atoms' dihedral angles into histogram bins
    hist, bnds = np.histogram(dihd_angls, bins=n_bins, range=(-np.pi, np.pi))
    prob = hist / np.sum(hist)

    # export statistics to a plain-text file
    with open(stt_fpath, 'w') as o_file:
        for idx_bin in range(n_bins):
            o_file.write('%.4f %.4f %.4f\n' % (bnds[idx_bin], bnds[idx_bin + 1], prob[idx_bin]))


if __name__ == '__main__':
    main()
