"""Common utility functions."""

import os
import json
import random
import logging
import hashlib
from datetime import datetime

import yaml


def tfold_init(path=None):
    """Initialize the tFold-xxx framework.

    Args:
    * (optional) path: path to the configuration YAML file

    Returns:
    * config: dict of configurations
    """

    # load configurations from the YAML file
    if path is not None and os.path.exists(path):
        with open(path, 'r') as i_file:
            config = yaml.safe_load(i_file)
    else:
        config = {'verbose_level': 'INFO'}  # default configurations

    # over-ride configurations with the additional JSON file
    jzw_path_key = 'JIZHI_WORKSPACE_PATH'
    if jzw_path_key in os.environ:
        jsn_fpath = os.path.join(os.getenv(jzw_path_key), 'job_param.json')
        if os.path.exists(jsn_fpath):
            with open(jsn_fpath, 'r') as i_file:
                jsn_data = json.load(i_file)
            for key, value in jsn_data.items():
                config[key] = value

    # configure the logging facility
    assert 'verbose_level' in config, '<verbose_level> not specified in the configuration YAML file'
    logging.basicConfig(
        format='[%(asctime)-15s %(levelname)s %(filename)s:L%(lineno)d] %(message)s',
        level=config['verbose_level'],
    )

    # display all the configuration items
    logging.info('tFold-SE3 initialized')
    for key, value in config.items():
        logging.info('%s => %s / %s', key, str(value), str(type(value)))

    return config


def get_md5sum(x_str, as_int=False):
    """Get the MD5 sum of the given string.

    Args:
    * x_str: input string
    * as_int: whether to return the MD5 sum as an integer

    Returns:
    * md5sum: MD5 sum string / integer
    """

    md5sum = hashlib.md5(x_str.encode('utf-8')).hexdigest()
    if as_int:
        md5sum = int(md5sum, 16)

    return md5sum


def get_rand_str():
    """Get a randomized string.

    Args: n/a

    Returns:
    * rand_str: randomized string
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    rand_val = random.random()
    rand_str_raw = '%s_%f' % (timestamp, rand_val)
    rand_str = hashlib.md5(rand_str_raw.encode('utf-8')).hexdigest()

    return rand_str


def get_nb_threads():
    """Get the number of parallel threads.

    Arg: n/a

    Returns:
    * nb_threads: number of parallel threads
    """

    if 'NB_THREADS' not in os.environ:
        nb_threads = 1
    else:
        nb_threads = int(os.getenv('NB_THREADS'))

    return nb_threads


def make_config_list(**kwargs):
    """Make a list of configurations from (key, list of values) pairs.

    Args:
    * kwargs: (key, list of values) pairs

    Returns:
    * config_list: list of configurations
    """

    config_list = []
    for key, values in kwargs.items():
        if not config_list:
            config_list = [{key: value} for value in values]
        else:
            config_list_new = []
            for config in config_list:
                config_list_new.extend([{**config, key: value} for value in values])
            config_list = config_list_new

    return config_list
