"""Mathematics-related utility functions."""

import numpy as np


def cvt_to_one_hot(arr, depth):
    """Convert an integer array into one-hot encodings.

    Args:
    * arr: integer array of size (D1, D2, ..., Dk)
    * depth: one-hot encodings's depth (denoted as C)

    Returns:
    * arr_oht: one-hot encodings of size (D1, D2, ..., Dk, C)
    """

    assert np.min(arr) >= 0 and np.max(arr) < depth
    arr_oht = np.reshape(np.eye(depth)[arr.ravel()], list(arr.shape) + [depth])

    return arr_oht


def get_rotate_mat():
    """Get a randomized 3D rotation matrix.

    Args: n/a

    Returns:
    * rotate_mat: 3D rotation matrix
    """

    # generate a randomized 3D rotation matrix
    yaw = np.random.uniform(-np.pi, np.pi)
    pitch = np.random.uniform(-np.pi, np.pi)
    roll = np.random.uniform(-np.pi, np.pi)
    sy, cy = np.sin(yaw), np.cos(yaw)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sr, cr = np.sin(roll), np.cos(roll)
    rotate_mat = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)

    return rotate_mat


def calc_plane_angle(cord_1, cord_2, cord_3):
    """Calculate the plane angle defined by 3 points' 3-D coordinates.

    Args:
    * cord_1: 3-D coordinate of the 1st point
    * cord_2: 3-D coordinate of the 2nd point
    * cord_3: 3-D coordinate of the 3rd point

    Returns:
    * rad: planar angle (in radian)
    """

    eps = 1e-6
    a1 = cord_1 - cord_2
    a2 = cord_3 - cord_2
    rad = np.arccos(np.clip(
        np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + eps), -1.0, 1.0))

    return rad


def calc_dihedral_angle(cord_1, cord_2, cord_3, cord_4):
    """Calculate the dihedral angle defined by 4 points' 3-D coordinates.

    Args:
    * cord_1: 3-D coordinate of the 1st point
    * cord_2: 3-D coordinate of the 2nd point
    * cord_3: 3-D coordinate of the 3rd point
    * cord_4: 3-D coordinate of the 4th point

    Returns:
    * rad: dihedral angle (in radian)
    """

    eps = 1e-6
    a1 = cord_2 - cord_1
    a2 = cord_3 - cord_2
    a3 = cord_4 - cord_3
    v1 = np.cross(a1, a2)
    v1 = v1 / np.sqrt((v1 * v1).sum(-1) + eps)
    v2 = np.cross(a2, a3)
    v2 = v2 / np.sqrt((v2 * v2).sum(-1) + eps)
    sign = np.sign((v1 * a3).sum(-1))
    rad = np.arccos(np.clip(
        (v1 * v2).sum(-1) / np.sqrt((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1) + eps), -1.0, 1.0))
    if sign != 0:
        rad *= sign

    return rad
