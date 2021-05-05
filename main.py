# Hi Jonas, my first change in Github
import numpy as np
np.set_printoptions(precision=4)  # print arrays to 4DP
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

#----------------------------------------------------------------------------
DATADIR = "Sample Data"         #Data directory
img_height = 64                 #Variables
img_width = 64
Img_dataset = []
Lab_dataset = []

#----------------------------------------------------------------------------
#Select augmentation here:

Aug_Whtlst = ["rotate", "scale", "flip", "translate", "blur"]
Aug_rand = rm.randrange(len(Aug_Whtlst))
print(Aug_rand)
print(Aug_Whtlst[Aug_rand])

#----------------------------------------------------------------------------
#load all nii image files from datadir in dataset
os.chdir(DATADIR + "/Image")
for nii in os.listdir(os.getcwd()):
    img = nib.load(nii)
    nii_data = img.get_fdata()
    #print(nii_data.shape)
    I = nii_data[..., 0]  # I is the first volume
    print(I.shape)
    Img_dataset.append(I)
Img_dataset = np.asarray(Img_dataset)

#load all nii segmentation files from datadir in dataset
os.chdir("..")                                          #go back one directory
os.chdir("Label")
for nii in os.listdir(os.getcwd()):
    lab = nib.load(nii)
    nii_data = lab.get_fdata()
    I = nii_data[..., 0]  # I is the first volume
    Lab_dataset.append(I)
    #print(Lab_dataset.shape)
Lab_dataset = np.asarray(Lab_dataset)
print("---------------------------------------")

if Img_dataset.shape != Lab_dataset.shape:                #if amount of images != amount of labels => error
    print("Something went wrong while reading data!")
    print(Img_dataset.shape)
    print(Lab_dataset.shape)
    exit()

#----------------------------------------------------------------------------
os.chdir("..")                                          #go back one directory

#augment given nii file
def augmentation(aug_int,file_int):
#-------------------------------------ROTATE-----------------------------------------------
    if aug_int == 0:
        print("---rotate---")
        for slice_Number in range(3):#range(nii_data.shape[2]):
            i = 10 #start at layer 10
            # rotation matrix for rotation of 0.2 radians around x axis
            M = y_rotmat(0.1)
            translation = [0,0,0]  # Translation from I to J [y,z,x]
            # order=1 for linear interpolation
            K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64,64,64), order=1)
            P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64,64,64), order=1)
            print(K.shape)
            plt.imshow(K[:, i])
            plt.show()
            i = i + 1

    # -------------------------------------SCALE-----------------------------------------------
    if aug_int == 1:
        print("---scale---")
        for slice_Number in range(3):#range(nii_data.shape[2]):
            i = 10  # start at layer 10
            # rotation matrix for rotation of 0.2 radians around x axis
            M = y_rotmat(0)
            translation = [0,0,0]  # Translation from I to J [y,z,x]
            # order=1 for linear interpolation
            K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(32,32,32), order=1)
            P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(32,32,32), order=1)
            print(K.shape)
            plt.imshow(K[:, i])
            plt.show()
            i = i + 1
    # -------------------------------------FLIP-----------------------------------------------
    if aug_int == 2:
        print("---flip---")
        for slice_Number in range(3):  # range(nii_data.shape[2]):
            i = 10  # start at layer 10
            K = np.flip(Img_dataset[file_int],1)
            P = np.flip(Lab_dataset[file_int],1)
            print(K.shape)
            plt.imshow(K[:, i])
            plt.show()
            i = i + 1
    # -------------------------------------TRANSLATE-----------------------------------------------
    if aug_int == 3:
        print("---translate---")
        for slice_Number in range(3):  # range(nii_data.shape[2]):
            i = 10  # start at layer 10
            translation = [-32.2, 24, -32.2]  # Translation from I to J [y,z,x]
            M = y_rotmat(0)
            # order=1 for linear interpolation
            K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64,64,64), order=1)
            P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(32,32,32), order=1)
            print(K.shape)
            plt.imshow(K[:, i])
            plt.show()
            i = i + 1
    # -------------------------------------SHIFT----------------------------------------------- (not working i guess)
    if aug_int == 4:
        print("---shift---")
        for slice_Number in range(3):#range(nii_data.shape[2]):
            i = 10  # start at layer 10
            # order=1 for linear interpolation
            K = shift(Img_dataset[file_int], 1,  order=1)
            P = shift(Lab_dataset[file_int], 1,  order=1)
            print(K.shape)
            plt.imshow(K[ :, i])
            plt.show()
            i = i + 1
    # -------------------------------------BLUR-----------------------------------------------
    if aug_int == 5:
        print("---blur---")
        for slice_Number in range(3):  # range(nii_data.shape[2]):
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

    new_image = nib.Nifti1Image(K, affine=np.eye(4))
    nib.save(new_image, 'augmentedImage.nii')
    new_label = nib.Nifti1Image(P, affine=np.eye(4))
    nib.save(new_label, 'augmentedLabel.nii')


augmentation(Aug_rand,1)




