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

np.set_printoptions(precision=4)  # print arrays to 4DP (float point precision, round to 4 digits)
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'  # gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib
import os
import random as rm
import Nifty_Augmentation_Config as augConfig
import Nifty_Augmentation as aug

# ----------------------------------------------------------------------------
DATADIR = "Sample Data"  # Data directory
Block_Size = [64, 64, 64]
Img_dataset = []
Lab_dataset = []
Img_str_lst = []  # Image string list
Lab_str_lst = []  # Label string list
augmentationFunctions = []
augmentationList = []
augmentationListMultiple = []

# ----------------------------------------------------------------------------
os.chdir(DATADIR + "/Image")  # change the current working directory to DATADIR
counter = 0
for nii in os.listdir(os.getcwd()):  # load all nii image files from datadir in dataset
    Img_str_lst.append(nii[:-4])  # append the images to a list without the file ending ( [:-4] removes the .nii)
    img = nib.load(nii)  # load nifty file
    nii_data = img.get_fdata()  # get the data as an float array
    I = nii_data[..., 0]  # I is the first volume
    print(I.shape)
    Img_dataset.append(I)
Img_dataset = np.asarray(Img_dataset)
print("------------------------------------")

os.chdir("..")  # go back one directory
os.chdir("Label")
for nii in os.listdir(os.getcwd()):  # load all nii segmentation files from datadir in dataset
    Lab_str_lst.append(nii[:-4])
    lab = nib.load(nii)
    nii_data = lab.get_fdata()
    I = nii_data[..., 0]  # I is the first volume
    Lab_dataset.append(I)
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


def augmentation_function_list_generation():
    """
    AUGMENTATION FUNCTION LIST GENERATION
    Brief:
    ------------
        Generates a List of all functions which are allowed to use from config file
        This function takes the Whitelist from the config file and generates a List of all callable functions which are
        allowed (set to True in whitelist).
        The List ist stored in the global variable augmentationFunction
    """
    global augmentationFunctions
    for x in augConfig.Aug_Whtlst:
        if augConfig.Aug_Whtlst[x]:
            augmentationFunctions.append(getattr(aug, x))


def augmentation_list_generation_multiple():
    """
    AUGMENTATION LIST GENERATION MULTIPLE
    Brief:
    ------------
        generates a list of callable functions in the from of an array like
        [
            [ aug1, aug8, aug4,...],
            [..., ...],
            [... ,... ,..]
        ]
        If a function is described but is not permitted to use, the function is skipped
    """
    global augmentationListMultiple
    i = 0
    for x in augConfig.Aug_Whtlst_Multiple_ordered:
        augmentationListMultiple.append([])  # append a new empty List element
        for y in x:
            func = getattr(aug, y)
            if func == AttributeError:  # Error Handling when the Function is configured as False in Whitelist
                print("ERROR: augmentation_list_generation_multiple:")
                print("\t" + y + "is skipped because it is not allowed")
            else:
                augmentationListMultiple[i].append(func)  # append the function out of the list names
        i = i + 1


def augmentation_multiple_ordered():
    """
    AUGMENTATION MULTIPLE ORDERED
    Brief:
    ------------
        augmentation of the filed as mentioned in multiple whitelist ordered
    """
    print ("multiple augmentation ordered")
    global augmentationListMultiple
    aug_str = ""
    sequence = rm.choice(augmentationListMultiple)  # get random sequenze from list
    for x in sequence:  # iterate through sequence with x
        aug_str_tmp = x()
        aug_str = aug_str + aug_str_tmp  # call in set sequence and store the name o the called function
    return aug_str


def augmentation_list_generation():
    """
    AUGMENTATION LIST GENERATION
    Brief:
    ------------
        creates the ready to use list, with single, multiple ordered, multiple unordered functions
        functions are inserted as many times as mentioned in the priority
         e.g if the customer likes more multiple augmentations then single augmentions
    """
    global augmentationList
    i = 0
    augmentation_function_list_generation()
    while i < augConfig.Single_Mode_Priority:
        i = i + 1
        augmentationList.extend(augmentationFunctions)

    if augConfig.Multiple_Mode_ordered_Active == 1:
        i = 0
        augmentation_list_generation_multiple()  # call function to convert the strings from config file to function
        # list
        while i < augConfig.Multiple_Mode_Ordered_Priority:
            i = i + 1
            augmentationList.append(augmentation_multiple_ordered)

    if augConfig.Multiple_Mode_unordered_Active == 1:
        i = 0
        while i < augConfig.Multiple_Mode_Unordered_Priority:
            i = i + 1
            augmentationList.append(augmentation_multiple_random_depth)  # append the random function in the array


def augmentation_multiple_random_depth():
    depth = rm.randrange(augConfig.Multiple_Aug_Depth_min, augConfig.Multiple_Aug_Depth_max, 1)
    print("multiple random depth: " , depth)
    x = 0
    aug_str = ""
    if depth < 0:
        while x < depth:
            x = x + 1
            aug_str_temp = rm.choice(augmentationFunctions)()
            aug_str = aug_str + aug_str_temp
    return aug_str


def augmentation_save(files, aug_str):
    new_image = nib.Nifti1Image(aug.K, affine=np.eye(4))
    nib.save(new_image, Img_str_lst[files] + '_' + aug_str + '.nii')
    new_label = nib.Nifti1Image(aug.P, affine=np.eye(4))
    nib.save(new_label, Lab_str_lst[files] + '_' + aug_str + '.nii')


def augmentation(image_dataset, lab_dataset):
    augmentation_list_generation()
    for files in range(len(image_dataset)):
        aug.K = image_dataset[files]
        aug.P = lab_dataset[files]
        # Separate here
        aug_str = rm.choice(augmentationList)()
        print(aug_str)
        augmentation_save(files, aug_str)



print(augmentationList)
augmentation(Img_dataset, Lab_dataset)
