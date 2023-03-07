"""Split domain IDs into training/validation/test subsets."""

import os
import logging

from tfold_se3.utils import tfold_init
from tfold_se3.utils import get_md5sum


def export_chain_ids(chain_ids, path):
    """Export chain IDs to a plain-text file."""

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w') as o_file:
        o_file.write('\n'.join(sorted(chain_ids)) + '\n')


def main():
    """Main entry."""

    # load configurations from the YAML file
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_fpath = os.path.join(curr_dir, 'config_gpu241.yaml')
    config = tfold_init(cfg_fpath)

    # get chain IDs
    with open(config['cid_fpath_all'], 'r') as i_file:
        chain_ids = [i_line.strip() for i_line in i_file]

    # split domain IDs into training/validation/test subsets
    n_chains = len(chain_ids)
    chain_ids.sort(key=get_md5sum)
    n_chains_trn = int(n_chains * config['ratio_trn'])
    n_chains_val = int(n_chains * config['ratio_val'])
    n_chains_tst = n_chains - n_chains_trn - n_chains_val
    logging.info('# of chains: %d (trn) / %d (val) / %d (tst)',
                 n_chains_trn, n_chains_val, n_chains_tst)
    chain_ids_trn = chain_ids[:n_chains_trn]
    chain_ids_val = chain_ids[n_chains_trn:-n_chains_tst]
    chain_ids_tst = chain_ids[-n_chains_tst:]

    # export domain IDs belonging to each subset to a plain-text file
    export_chain_ids(chain_ids_trn, config['cid_fpath_trn'])
    export_chain_ids(chain_ids_val, config['cid_fpath_val'])
    export_chain_ids(chain_ids_tst, config['cid_fpath_tst'])


if __name__ == '__main__':
    main()
