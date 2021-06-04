import random as rm
from rotations import x_rotmat  # from rotations.py
from rotations import y_rotmat  # from rotations.py
from rotations import z_rotmat  # from rotations.py
# ----------------------------------------------------------------------------
# random axis roations
# this function is called when a random axis rotation should be done
rot_array = [x_rotmat, y_rotmat, z_rotmat]


def rotmat(matrix):
    random_axis = rm.randrange(0, 3, 1)
    print("Ration: ", random_axis)
    if random_axis == 0:
        print("x_rotation")

    elif random_axis == 1:
        print("y_rotation")

    elif (random_axis) == 2:
        print("z_rotation")

    return rot_array[random_axis](matrix)