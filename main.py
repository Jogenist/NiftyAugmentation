"""
main.py

This file initiates the nifty augmentation.
First it reads all nifty-images and segmentations as numpy arrays.
Then it randomly applies augmentations from Nifty_Augmentation.py to the files and saves them as new a nifty.

06/2021
"""

# ----------------------------------------------------------------------------
# import libraries
import numpy as np
np.set_printoptions(precision=4)                                                                                        # print arrays to 4DP (float point precision, round to 4 digits)
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'                                                                                     # gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib
import os
import random as rm
import Nifty_Augmentation_Config as augConfig
import Nifty_Augmentation as aug

# ----------------------------------------------------------------------------
DATADIR = "Sample Data"                                                                                                 # Data directory
Block_Size = [64, 64, 64]
Img_dataset = []
Lab_dataset = []
Img_str_lst = []                                                                                                        # Image string list
Lab_str_lst = []                                                                                                        # Label string list
augmentationList = []

# ----------------------------------------------------------------------------
os.chdir(DATADIR + "/Image")                                                                                            # change the current working directory to DATADIR
counter = 0
for nii in os.listdir(os.getcwd()):                                                                                     # load all nii image files from datadir in dataset
    Img_str_lst.append(nii[:-4])                                                                                        # append the images to a list without the file ending ( [:-4] removes the .nii)
    img = nib.load(nii)                                                                                                 # load nifty file
    nii_data = img.get_fdata()                                                                                          # get the data as an float array
    I = nii_data[..., 0]                                                                                                # I is the first volume
    print(I.shape)
    Img_dataset.append(I)
Img_dataset = np.asarray(Img_dataset)
print("------------------------------------")


os.chdir("..")                                                                                                          # go back one directory
os.chdir("Label")
for nii in os.listdir(os.getcwd()):                                                                                     # load all nii segmentation files from datadir in dataset
    Lab_str_lst.append(nii[:-4])
    lab = nib.load(nii)
    nii_data = lab.get_fdata()
    I = nii_data[..., 0]                                                                                                # I is the first volume
    Lab_dataset.append(I)
Lab_dataset = np.asarray(Lab_dataset)
print("---------------------------------------")

if Img_dataset.shape != Lab_dataset.shape:                                                                              # if amount of images != amount of labels => error
    print("Something went wrong while reading data!")
    print(Img_dataset.shape)
    print(Lab_dataset.shape)
    exit()

# ----------------------------------------------------------------------------
os.chdir("..")                                                                                                          # go back one directory

# ----------------------------------------------------------------------------


def augmentation_list():                                                                                                # Generates a List of all functions which are allowed to use from config file
    global augmentationList
    for x in augConfig.Aug_Whtlst:
        if augConfig.Aug_Whtlst[x]:
            augmentationList.append(getattr(aug, x))


def augmentation(image_dataset, lab_dataset):

    for files in range(len(image_dataset)):
        aug
        file_int, aug_str = rm.choice(augmentationList)(image_dataset,lab_dataset,files)
        print(aug_str)
        new_image = nib.Nifti1Image(aug.K, affine=np.eye(4))
        nib.save(new_image, Img_str_lst[file_int] + '_' + aug_str + '.nii')
        new_label = nib.Nifti1Image(aug.P, affine=np.eye(4))
        nib.save(new_label, Lab_str_lst[file_int] + '_' + aug_str + '.nii')


augmentation_list()
augmentation(Img_dataset, Lab_dataset)



