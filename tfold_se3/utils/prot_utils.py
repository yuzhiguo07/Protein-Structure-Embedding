"""Protein-related utility functions."""

import os

import numpy as np


# constants
AA_NAMES_DICT_1TO3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS',
    'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN',
    'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}
AA_NAMES_DICT_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_NAMES_1CHAR = 'ACDEFGHIKLMNPQRSTVWY'


def parse_fas_file(path):
    """Parse the FASTA file.

    Args:
    * path: path to the FASTA file

    Returns:
    * prot_id: protein ID (as in the commentary line)
    * aa_seq: amino-acid sequence
    """

    assert os.path.exists(path), 'FASTA file does not exist: ' + path
    with open(path, 'r') as i_file:
        i_lines = [i_line.strip() for i_line in i_file]
        prot_id = i_lines[0][1:]
        aa_seq = ''.join(i_lines[1:])

    return prot_id, aa_seq

def parse_pdb_file(path, atom_name='CA'):
    """Parse the PDB file to obtain atom coordinates.

    Args:
    * path: path to the PDB file

    Returns:
    * cord_mat: atoms' 3D coordinates (N x 3)
    * mask_vec: atoms' validness masks (N)
    """

    # check whether the PDB file exists
    assert os.path.exists(path), 'PDB file does not exist: ' + path

    # parse the PDB file
    idxs_resd = set()
    atom_dict = {}
    with open(path, 'r') as i_file:
        for i_line in i_file:
            if not i_line.startswith('ATOM'):
                continue
            idx_resd = int(i_line[22:26])
            idxs_resd.add(idx_resd)
            if i_line[12:16].strip() == atom_name:
                cord_x = float(i_line[30:38])
                cord_y = float(i_line[38:46])
                cord_z = float(i_line[46:54])
                atom_dict[idx_resd] = np.array([cord_x, cord_y, cord_z], dtype=np.float32)

    # extract the coordinate matrix & mask vector
    idxs_resd = sorted(list(idxs_resd))
    cord_mat = np.stack([
        atom_dict.get(x, np.zeros((3), dtype=np.float32)) for x in idxs_resd
    ], axis=0)
    mask_vec = np.array([1 if x in atom_dict else 0 for x in idxs_resd], dtype=np.int8)

    return cord_mat, mask_vec

def export_fas_file(prot_id, aa_seq, path):
    """Export the amino-acid sequence to a FASTA file.

    Args:
    * prot_id: protein ID (as in the commentary line)
    * aa_seq: amino-acid sequence
    * path: path to the FASTA file

    Returns: n/a
    """

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        o_file.write('>%s\n%s\n' % (prot_id, aa_seq))

def export_pdb_file(aa_seq, atom_cords, path, atom_masks=None):
    """Export the 3D structure to a PDB file.

    Args:
    * aa_seq: amino-acid sequence
    * atom_cords: atom coordinates (['CA']: L x 3 / ['N', 'CA', 'C']: L x 3 x 3)
    * path: path to the PDB file
    * (optional) atom_masks: atom masks (['CA']: L / ['N', 'CA', 'C']: L x 3)

    Returns: n/a
    """

    # configurations
    alt_loc = ' '
    chain_id = 'A'
    i_code = ' '
    occupancy = 1.0
    temp_factor = 1.0
    element = 'C'
    charge = ' '
    cord_min = -999.0
    cord_max = 999.0

    # validate input arguments
    assert len(aa_seq) == atom_cords.shape[0]
    if atom_masks is not None:
        assert len(aa_seq) == atom_masks.shape[0]
    else:
        atom_masks = np.ones(atom_cords.shape[:-1], dtype=np.int8)

    # determine the set of atom names (per residue)
    if atom_cords.ndim == 2:
        atom_names = ['CA']
        atom_cords_ext = atom_cords[:, None, :]
        atom_masks_ext = atom_masks[:, None]
    else:
        atom_names = ['N', 'CA', 'C']
        atom_cords_ext = atom_cords
        atom_masks_ext = atom_masks

    # reset invalid values in atom coordinates
    atom_cords_ext = np.clip(atom_cords_ext, cord_min, cord_max)
    atom_cords_ext[np.isnan(atom_cords_ext)] = 0.0
    atom_cords_ext[np.isinf(atom_cords_ext)] = 0.0

    # export the 3D structure to a PDB file
    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        n_atoms = 0
        for idx_resd, resd_name in enumerate(aa_seq):
            for idx_atom, atom_name in enumerate(atom_names):
                if atom_masks_ext[idx_resd, idx_atom] == 0:
                    continue
                n_atoms += 1
                line_str = 'ATOM  '
                line_str += '%5d' % n_atoms
                line_str += '  ' + atom_name + ' ' * (3 - len(atom_name))
                line_str += alt_loc
                line_str += '%3s' % AA_NAMES_DICT_1TO3[resd_name]
                line_str += ' %s' % chain_id
                line_str += '%4d' % (idx_resd + 1)
                line_str += '%s   ' % i_code
                line_str += '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 0]
                line_str += '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 1]
                line_str += '%8.3f' % atom_cords_ext[idx_resd, idx_atom, 2]
                line_str += '%6.2f' % occupancy
                line_str += '%6.2f' % temp_factor
                line_str += ' ' * 10
                line_str += '%2s' % element
                line_str += '%2s' % charge
                assert len(line_str) == 80, line_str
                o_file.write(line_str + '\n')
