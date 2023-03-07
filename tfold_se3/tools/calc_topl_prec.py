"""Calculate top-L precision of long-range contacts."""

import argparse

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from tfold_se3.utils import parse_fas_file
from tfold_se3.utils import parse_pdb_file


def get_cntcs_true(path):
    """Get ground-truth contacting residue pairs."""

    cord_mat, mask_vec = parse_pdb_file(path, atom_name='CB')
    dist_mat = squareform(pdist(cord_mat, metric='euclidean'))
    mask_mat = (dist_mat <= 8.0).astype(np.int8) * mask_vec[:, None] * mask_vec[None, :]

    return mask_mat


def get_cntcs_pred(fas_fpath, npz_fpath):
    """Get predicted contacting residue pairs."""

    # parse the FASTA file
    _, aa_seq = parse_fas_file(fas_fpath)
    seq_len = len(aa_seq)

    # parse the NPZ file
    with np.load(npz_fpath) as npz_data:
        dist_tns = npz_data['dist']
        prob_mat = np.sum(dist_tns[:, :, 1:13], axis=-1)
        assert prob_mat.shape[0] == seq_len

    # get predicted contacting residue pairs
    cntc_infos = []
    for ir in range(seq_len):
        for ic in range(ir + 24, seq_len):
            if aa_seq[ir] == 'G' or aa_seq[ic] == 'G':
                continue
            cntc_infos.append((ir, ic, prob_mat[ir, ic]))
    cntc_infos.sort(key=lambda x: x[2], reverse=True)
    cntc_infos = cntc_infos[:min(seq_len, len(cntc_infos))]  # only keep top-L pairs

    return cntc_infos


def calc_topl_prec(fas_fpath, pdb_fpath, npz_fpath):
    """Calculate top-L precision of long-range contacts."""

    mask_mat = get_cntcs_true(pdb_fpath)
    cntc_infos = get_cntcs_pred(fas_fpath, npz_fpath)

    nb_pairs_full = len(cntc_infos)
    nb_pairs_true = 0
    for ir, ic, _ in cntc_infos:
        if mask_mat[ir, ic] == 1:
            nb_pairs_true += 1
    prec = (nb_pairs_true / nb_pairs_full) if nb_pairs_full > 0 else 0.0

    return prec


def main():
    """Main entry."""

    # parse input arguments
    parser = argparse.ArgumentParser(description='Calculate top-L precision of long-range contacts')
    parser.add_argument('--fas_fpath', type=str, required=True, help='path to the FASTA file')
    parser.add_argument('--pdb_fpath', type=str, required=True, help='path to the native PDB file')
    parser.add_argument('--npz_fpath', type=str, required=True, help='path to the NPZ file')
    args = parser.parse_args()

    # calculate the top-L precision of long-range contacts
    prec = calc_topl_prec(args.fas_fpath, args.pdb_fpath, args.npz_fpath)
    print('Top-L precision: %.4f' % prec)

if __name__ == '__main__':
    main()
