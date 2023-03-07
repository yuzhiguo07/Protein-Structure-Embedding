"""PyTorch-related utility functions."""

import logging

import torch


def get_tensor_size(tensor):
    """Get the PyTorch tensor's memory consumption (in MB).

    Args:
    * tensor: PyTorch tensor

    Returns:
    * mem: PyTorch tensor's memory consumption (in MB)
    """

    mem = tensor.element_size() * torch.numel(tensor) / 1024.0 / 1024.0

    return mem


def check_tensor_size(tensor, name, mem_thres=500.0):
    """Check whether the PyTorch tensor consumes more memory than the threshold.

    Args:
    * tensor: PyTorch tensor
    * name: PyTorch tensor's name
    * mem_thres: memory consumption's threshold (in MB)

    Returns: n/a
    """

    mem = get_tensor_size(tensor)
    if mem > mem_thres:
        logging.debug('tensor <%s> has consumed %.2f MB memory', name, mem)


def get_peak_memory():
    """Get the peak memory consumption (in GB).

    Args: n/a

    Returns:
    * mem: peak memory consumption (in GB)
    """

    mem = torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024.0 / 1024.0

    return mem
