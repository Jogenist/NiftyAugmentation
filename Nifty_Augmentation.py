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

# -------------------------------------ROTATE-----------------------------------------------
def rotate(file_int):
    print("---rotate---")

    # -------------------------------------SCALE-----------------------------------------------


def scale(file_int):
    print("---scale---")


# -------------------------------------FLIP-----------------------------------------------
def flip(file_int):
    print("---flip---")



# -------------------------------------TRANSLATE-----------------------------------------------
def translate(file_int):
    print("---translate---")



# -------------------------------------SKEW-----------------------------------------------
def skew(file_int):
    print("---skew---")


# -------------------------------------BLUR-----------------------------------------------
def blur(file_int):
    print("---blur---")



# -------------------------------------RANDOM CROP AND RESIZE-----------------------------------------------
def cropAndResize(file_int):
    print("---Crop & Resize---")


# -------------------------------------CROP AND PATCH-----------------------------------------------
def cropAndPatch(file_int):
    print("---Crop & Patch---")


# -------------------------------------ELASTIC DISTORTION-----------------------------------------------
def elasticDistortion(file_int):
    print("---Elastic Distortion---")


# -------------------------------------RANDOM ERASING-----------------------------------------------
def randomErasing(file_int):
    print("---Random Erasing---")


# -------------------------------------NOISE-----------------------------------------------
def noise(file_int):
    print("---noise---")


# -------------------------------------SCALE-----------------------------------------------
def shear(file_int):
    print("---shear---")


# -------------------------------------SALT & PEPPER-----------------------------------------------
def saltAndPepper(file_int):
    print("---Salt and Pepper---")

def noAugmentation(file_int):
    print("---no augmentation---")

