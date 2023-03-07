"""Main entry for training and sampling with <EbmLearner>."""

import os
import argparse

from tfold_se3.utils import tfold_init
from tfold_se3.experiments.ebm_struct.ebm_learner import EbmLearner


def main():
    """Main entry."""
    parser = argparse.ArgumentParser(description='run ebm model')
    parser.add_argument('--config_fname', required=True, help='config*.yaml file name')
    args = parser.parse_args()


    # initialize the tFold-SE3 framework
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    # yml_fpath = os.path.join(curr_dir, 'config.yaml')
    # yml_fpath = os.path.join(curr_dir, 'config_loc.yaml')
    yml_fpath = os.path.join(curr_dir, args.config_fname)
    config = tfold_init(yml_fpath)

    # train a EBM model or sample 3D structure with this model
    learner = EbmLearner(config)
    if config['exec_mode'] == 'train':
        learner.train()
    elif config['exec_mode'] == 'sample':
        learner.sample()
    elif config['exec_mode'] == 'finetune':
        learner.finetune()
    elif config['exec_mode'] == 'sample_feat':
        learner.sample_feat()
    else:
        raise ValueError('unrecognized execution mode: %s' % config['exec_mode'])


if __name__ == '__main__':
    main()
