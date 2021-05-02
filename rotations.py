""" Rotation matrices for rotations around x, y, z axes

See: https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations
"""

import numpy as np
import math


def x_rotmat(theta):
    """ Rotation matrix for rotation of `theta` radians around x axis

    Parameters
    ----------
    theta : scalar
        Rotation angle in radians

    Returns
    -------
    M : shape (3, 3) array
        Rotation matrix
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, cos_t, -sin_t],
                     [0, sin_t, cos_t]])


def y_rotmat(theta):
    """ Rotation matrix for rotation of `theta` radians around y axis

    Parameters
    ----------
    theta : scalar
        Rotation angle in radians

    Returns
    -------
    M : shape (3, 3) array
        Rotation matrix
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([[cos_t, 0, sin_t],
                     [0, 1, 0],
                     [-sin_t, 0, cos_t]])


def z_rotmat(theta):
    """ Rotation matrix for rotation of `theta` radians around z axis

    Parameters
    ----------
    theta : scalar
        Rotation angle in radians

    Returns
    -------
    M : shape (3, 3) array
        Rotation matrix
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([[cos_t, -sin_t, 0],
                     [sin_t, cos_t, 0],
                     [0, 0, 1]])


def shear(angle, x, y):
    '''
    |1  -tan(𝜃/2) |  |1        0|  |1  -tan(𝜃/2) |
    |0      1     |  |sin(𝜃)   1|  |0      1     |
    '''
    # shear 1
    tangent = math.tan(angle / 2)
    new_x = round(x - y * tangent)
    new_y = y

    # shear 2
    new_y = round(new_x * math.sin(angle) + new_y)  # since there is no change in new_x according to the shear matrix

    # shear 3
    new_x = round(new_x - new_y * tangent)  # since there is no change in new_y according to the shear matrix

    return new_y, new_x
