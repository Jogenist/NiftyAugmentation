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

import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'  # gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib
import os
import Nifty_Augmentation as niiAug

# ----------------------------------------------------------------------------
DATADIR = "Sample Data"  # Data directory


# ----------------------------------------------------------------------------
os.chdir(DATADIR + "/Image")  # change the current working directory to DATADIR
counter = 0
for nii in os.listdir(os.getcwd()):  # load all nii image files from datadir in dataset
    niiAug.dataset.Img_str_lst.append(nii[:-4])  # append the images to a list without the file ending ( [:-4] removes the .nii)
    print(niiAug.dataset.Img_str_lst)
    img = nib.load(nii)  # load nifty file
    nii_data = img.get_fdata()  # get the data as an float array
    I = nii_data[..., 0]  # I is the first volume
    print(I.shape)
    niiAug.dataset.Img.append(I)
niiAug.dataset.Img = np.asarray(niiAug.dataset.Img)
print("------------------------------------")

os.chdir("..")  # go back one directory
os.chdir("Label")
for nii in os.listdir(os.getcwd()):  # load all nii segmentation files from datadir in dataset
    niiAug.dataset.Lab_str_lst.append(nii[:-4])
    lab = nib.load(nii)
    nii_data = lab.get_fdata()
    I = nii_data[..., 0]  # I is the first volume
    niiAug.dataset.Lab.append(I)
niiAug.dataset.Lab = np.asarray(niiAug.dataset.Lab)
print("---------------------------------------")
# ----------------------------------------------------------------------------
# check dataset
niiAug.dataset.self_check()


# ----------------------------------------------------------------------------
os.chdir("..")  # go back one directory


# ----------------------------------------------------------------------------
# Augment the Dataset
niiAug.augmentation(niiAug.dataset.Img, niiAug.dataset.Lab)
