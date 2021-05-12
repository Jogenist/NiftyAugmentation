import numpy as np
# Comment
np.set_printoptions(precision=4)  # print arrays to 4DP (float point precision, round to 4 digits)
import matplotlib.pyplot as plt

# gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
from scipy.ndimage import affine_transform
from scipy.ndimage import shift
from rotations import x_rotmat  # from rotations.py
from rotations import y_rotmat  # from rotations.py
from rotations import z_rotmat  # from rotations.py
import nibabel as nib
import os
import random as rm
from random import randrange

# ----------------------------------------------------------------------------
DATADIR = "Sample Data"  # Data directory
img_height = 64  # Variables
img_width = 64
Img_dataset = []
Lab_dataset = []
Img_str_lst = []  # Image Structure List?
Lab_str_lst = []  # Laboratory Structure List?

# ----------------------------------------------------------------------------
# Select augmentation here:

Aug_Whtlst = ["rotate", "scale", "flip", "translate", "blur", "no augmentation"]

# ----------------------------------------------------------------------------
# load all nii image files from datadir in dataset
os.chdir(DATADIR + "/Image")  # change the current working directory to DATADIR
counter = 0
for nii in os.listdir(os.getcwd()):
    Img_str_lst.append(nii[:-4])  # append tha images to a list without the file ending ( [:-4] removes the .nii)
    img = nib.load(nii)  # load nifty file
    nii_data = img.get_fdata()  # get the data as an float array
    # print(nii_data)
    I = nii_data[..., 0]  # I is the first volume
    # print(I)
    Img_dataset.append(I)
Img_dataset = np.asarray(Img_dataset)

# load all nii segmentation files from datadir in dataset
os.chdir("..")  # go back one directory
os.chdir("Label")
for nii in os.listdir(os.getcwd()):
    Lab_str_lst.append(nii[:-4])
    lab = nib.load(nii)
    nii_data = lab.get_fdata()
    I = nii_data[..., 0]  # I is the first volume
    Lab_dataset.append(I)
    # print(Lab_dataset.shape)
Lab_dataset = np.asarray(Lab_dataset)
print("---------------------------------------")

if Img_dataset.shape != Lab_dataset.shape:  # if amount of images != amount of labels => error
    print("Something went wrong while reading data!")
    print(Img_dataset.shape)
    print(Lab_dataset.shape)
    exit()

# ----------------------------------------------------------------------------
os.chdir("..")  # go back one directory

# ----------------------------------------------------------------------------
# random axis roations
# this function is called when a random axis roatation should be done
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


# ----------------------------------------------------------------------------

# augment given nii file
def augmentation(aug_int, file_int):
    # -------------------------------------ROTATE-----------------------------------------------
    if aug_int == 0:
        print("---rotate---")
        angle = rm.uniform(0.1, 0.5)
        print(angle)
        i = 10  # start at layer 10
        # rotation matrix for rotation of 0.2 radians around x axis
        M = rotmat(angle)
        translation = [0, 0, 0]  # Translation from I to J [y,z,x]
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "rotated"

    # -------------------------------------SCALE-----------------------------------------------
    if aug_int == 1:
        print("---scale---")
        i = 10  # start at layer 10
        # rotation matrix for rotation of 0.2 radians around x axis
        M = rotmat(0)
        translation = [0, 0, 0]  # Translation from I to J [y,z,x]
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(32, 32, 32), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(32, 32, 32), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "scaled"
    # -------------------------------------FLIP-----------------------------------------------
    if aug_int == 2:
        print("---flip---")
        i = 10  # start at layer 10
        K = np.flip(Img_dataset[file_int], 1)
        P = np.flip(Lab_dataset[file_int], 1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "flipped"
    # -------------------------------------TRANSLATE-----------------------------------------------
    if aug_int == 3:
        print("---translate---")
        i = 10  # start at layer 10
        translation = [-32.2, 24, -32.2]  # Translation from I to J [y,z,x]
        M = y_rotmat(0)
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(32, 32, 32), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "translated"
    # -------------------------------------SHIFT----------------------------------------------- (not working i guess)
    if aug_int == 10:  # shift is not working, therefore it cannot be reached as random augmentation
        print("---shift---")
        i = 10  # start at layer 10
        # order=1 for linear interpolation
        K = shift(Img_dataset[file_int], 1, order=1)
        P = shift(Lab_dataset[file_int], 1, order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "shifted"
    # -------------------------------------BLUR-----------------------------------------------
    if aug_int == 4:
        print("---blur---")
        i = 10  # start at layer 10
        translation = [-0.5, 0, 0.5]  # Translation from I to J [y,z,x]
        M = y_rotmat(0)
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "blured"
    # -------------------------------------NO AUGMENTATION-----------------------------------------------
    if aug_int == 5:
        print("---no augmentation---")
        i = 10  # start at layer 10
        K = Img_dataset[file_int]
        P = Lab_dataset[file_int]
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        return ()  # dont save unaugmented image, therefore leave here

    new_image = nib.Nifti1Image(K, affine=np.eye(4))
    nib.save(new_image, Img_str_lst[file_int] + '_' + aug_str + '.nii')
    new_label = nib.Nifti1Image(P, affine=np.eye(4))
    nib.save(new_label, Lab_str_lst[file_int] + '_' + aug_str + '.nii')


for files in range(len(Img_dataset)):
    Aug_rand = rm.randrange(len(Aug_Whtlst))
    print(Aug_rand)
    print(Aug_Whtlst[Aug_rand])
    print(files)
    augmentation(Aug_rand, files)
