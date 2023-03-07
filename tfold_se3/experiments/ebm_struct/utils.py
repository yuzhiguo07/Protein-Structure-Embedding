"""Utility functions."""

import subprocess

import numpy as np


def calc_noise_stds(noise_std_beg, noise_std_end, n_noise_levls):
    """Calculate a series of random noise's standard deviations."""

    base = 2.0  # equivalent for any values larger than 1
    noise_std_beg_log = np.log(noise_std_beg) / np.log(base)
    noise_std_end_log = np.log(noise_std_end) / np.log(base)
    noise_stds = np.logspace(
        noise_std_beg_log, noise_std_end_log, num=n_noise_levls, base=base, dtype=np.float32)

    return noise_stds


def calc_gdt_ts(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the GDT-TS score."""

    cmd_out = subprocess.check_output(['DeepScore', pdb_fpath_mod, pdb_fpath_ref, '-P 0 -n -2'])
    line_str = cmd_out.decode('utf-8')
    gdt_ts = float(line_str.split()[14])

    return gdt_ts


def calc_lddt_ca(pdb_fpath_mod, pdb_fpath_ref):
    """Calculate the lDDT-Ca score."""

    cmd_out = subprocess.check_output(['lddt', '-c', pdb_fpath_mod, pdb_fpath_ref])
    line_strs = cmd_out.decode('utf-8').split('\n')
    for line_str in line_strs:
        if line_str.startswith('Global LDDT score'):
            lddt_ca = float(line_str.split()[-1])

    return lddt_ca


def project_3d_cords(cord_mat_src, ca_dist=3.8):
    """Project 3D coordinates so that adjacent CA-CA distance is preserved."""

    seq_len = cord_mat_src.shape[0]
    dist_vec = np.linalg.norm(cord_mat_src[:-1] - cord_mat_src[1:], axis=-1)
    cord_mat_dst = np.copy(cord_mat_src)
    for idx in range(seq_len - 1, 0, -1):  # no need to adjust the first atom
        alpha = ca_dist / dist_vec[idx - 1]
        cord_mat_dst[idx:] += (alpha - 1.0) * (cord_mat_dst[idx] - cord_mat_dst[idx - 1])
    cord_mat_dst -= np.mean(cord_mat_dst, axis=0, keepdims=True)

    return cord_mat_dst
