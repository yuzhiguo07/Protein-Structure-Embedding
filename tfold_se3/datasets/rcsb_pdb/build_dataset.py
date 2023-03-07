"""Build the dataset (stored as HDF5 files) from raw RCSB-PDB files."""

import os
import re
import random
import logging
from multiprocessing import Pool

import h5py
from Bio.PDB.PDBExceptions import PDBConstructionException

from tfold_se3.utils import tfold_init
from tfold_se3.utils import get_nb_threads
from tfold_se3.utils import recreate_directory
from tfold_se3.tools.pdb_parser import PdbParser
from tfold_se3.tools.pdb_parser import PdbParserStatus


def parse_bcx_data(path):
    """Parse BC-X sequence clustering results."""
    pdb_chain_name = False

    # parse BC-X sequence clustering results
    bcx_data = []
    regex = re.compile(r'^[0-9A-Z]{4}_[A-Z]$')
    with open(path, 'r') as i_file:
        for i_line in i_file:
            if pdb_chain_name:
                chain_names = [x for x in i_line.split() if re.search(regex, x)]
                if len(chain_names) > 0:
                    bcx_data.append(chain_names)
            else:
                bcx_data.append(i_line.strip())
    random.shuffle(bcx_data)

    # show BC-X data statistics
    n_clsts = len(bcx_data)
    clst_size = sum([len(x) for x in bcx_data]) / n_clsts
    logging.info('# of BC-X clusters: %d', n_clsts)
    logging.info('# of chains per BC-X cluster: %.2f', clst_size)

    return bcx_data


def build_hdf_file(bcx_data, pdb_dpath, hdf_fpath, log_fpath):
    """Build HDF5 files based on BC-X sequence clustering results."""

    # show the greeting message
    logging.info('generating the HDF5 file: %s', hdf_fpath)

    # use the first successfully processed chain for each BC-X cluster
    pdb_data = {}
    status_list = []
    parser = PdbParser()
    for chain_names in bcx_data:
        status = None
        for chain_name in chain_names:
            pdb_code, chain_id = chain_name.split('_')
            pgz_fpath = os.path.join(
                pdb_dpath, pdb_code[1:-1].lower(), 'pdb%s.ent.gz' % pdb_code.lower())
            try:
                aa_seq, atom_cords, atom_masks, status = parser.run(pgz_fpath, chain_id)
                status_list.append((pgz_fpath, chain_id, status))
            except PDBConstructionException as pdb_error:
                raise PDBConstructionException('PDB: %s-%s' % (pgz_fpath, chain_id)) from pdb_error
            if status == PdbParserStatus.NORMAL:
                pdb_data[chain_name] = (aa_seq, atom_cords, atom_masks)
                break

    # export the PDB data to a HDF5 file
    os.makedirs(os.path.dirname(os.path.realpath(hdf_fpath)), exist_ok=True)
    with h5py.File(hdf_fpath, 'w') as o_file:
        for chain_name, (aa_seq, atom_cords, atom_masks) in pdb_data.items():
            group = o_file.create_group(chain_name)
            group.create_dataset('aa_seq', data=aa_seq)
            group.create_dataset('atom_cords', data=atom_cords)
            group.create_dataset('atom_masks', data=atom_masks)

    # export PDB parsing statistics to a LOG file
    os.makedirs(os.path.dirname(os.path.realpath(log_fpath)), exist_ok=True)
    with open(log_fpath, 'w') as o_file:
        for pgz_fpath, chain_id, status in status_list:
            o_file.write('%s / %s / %s\n' % (pgz_fpath, chain_id, str(status)))


def main():
    """Main entry."""

    # load configurations from the YAML file
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_fpath = os.path.join(curr_dir, 'config.yaml')
    config = tfold_init(cfg_fpath)

    # additional configurations
    n_threads = get_nb_threads()
    bcx_fpath = os.path.join(config['bcx_dpath'], 'bc-%d.out' % config['bcx_thres'])

    # remove previously generated HDF5 files
    recreate_directory(config['hdf_dpath'])
    recreate_directory(config['log_dpath'])

    # parse BC-X sequence clustering results
    bcx_data = parse_bcx_data(bcx_fpath)
    if config['n_clsts_max'] != -1 and len(bcx_data) > config['n_clsts_max']:
        logging.info('selecting %d out of %d BC-X clusters', config['n_clsts_max'], len(bcx_data))
        bcx_data = random.sample(bcx_data, config['n_clsts_max'])

    # build HDF5 files based on BC-X sequence clustering results
    n_clsts = len(bcx_data)
    n_files = (n_clsts + config['n_chains_per_file'] - 1) // config['n_chains_per_file']
    logging.info('building %d HDF5 files from %s BC-X clusters', n_files, n_clsts)
    args_list = []
    for idx_file in range(n_files):
        idx_clst_beg = config['n_chains_per_file'] * idx_file
        idx_clst_end = min(n_clsts, idx_clst_beg + config['n_chains_per_file'])
        bcx_data_sel = bcx_data[idx_clst_beg:idx_clst_end]
        hdf_fname = 'bc%d-%04d-of-%04d.hdf5' % (config['bcx_thres'], idx_file, n_files)
        hdf_fpath = os.path.join(config['hdf_dpath'], hdf_fname)
        log_fpath = os.path.join(config['log_dpath'], hdf_fname.replace('.hdf5', '.log'))
        args_list.append((bcx_data_sel, config['pdb_dpath_raw'], hdf_fpath, log_fpath))
    with Pool(processes=n_threads) as pool:
        pool.starmap(build_hdf_file, args_list)

if __name__ == '__main__':
    main()
