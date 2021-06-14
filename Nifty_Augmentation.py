"""
Nifty_Augmentation.py

Contains functions and implementation for serveral augmentation methods. Is used in main.py to augment Nifty-Files.
Uses Nifty_Augmentation_Config for user-customizable augmentation parameters.

06/2021
"""
# ----------------------------------------------------------------------------
# Import
import numpy as np
import nibabel as nib
import random as rm
import Nifty_Augmentation_Config as augConfig
import Nifty_Augmentation_functions as augFunc

# ----------------------------------------------------------------------------
# Defines
np.set_printoptions(precision=4)  # print arrays to 4DP (float point precision, round to 4 digits)
augmentationFunctions = []
augmentationList = []
augmentationListMultiple = []


# ----------------------------------------------------------------------------
# File Handler
class NiftyImage:
    def __init__(self):
        self.Img = []
        self.Lab = []
        self.Img_str_lst = []
        self.Lab_str_lst = []

    def self_check(self):
        if self.Img.shape != self.Lab.shape:  # if amount of images != amount of labels => error
            print("Something went wrong while reading data!")
            print(self.Img.shape)
            print(self.Lab.shape)
            exit()

    # Img = []
    # Lab = []
    # Img_str_lst = []  # Image string list
    # Lab_str_lst = []  # Label string list


dataset = NiftyImage()


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
            augmentationFunctions.append(getattr(augFunc, x))


# ----------------------------------------------------------------------------

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
            func = getattr(augFunc, y)
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
    print("multiple augmentation ordered")
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
    print("--- generate List from Whitelist ---")
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
    print("multiple random depth: ", depth)
    x = 0
    aug_str = ""
    if depth > 0:
        while x < depth:
            x = x + 1
            aug_str_temp = rm.choice(augmentationFunctions)()
            aug_str = aug_str + aug_str_temp
            print("multiple: " + aug_str)
    return aug_str


def augmentation_save(files, aug_str):
    new_image = nib.Nifti1Image(augFunc.K, affine=np.eye(4))
    nib.save(new_image, dataset.Img_str_lst[files] + '_' + aug_str + '.nii')
    new_label = nib.Nifti1Image(augFunc.P, affine=np.eye(4))
    nib.save(new_label, dataset.Lab_str_lst[files] + '_' + aug_str + '.nii')


def augmentation(image_dataset, lab_dataset):
    augmentation_list_generation()
    counter = 0
    for files in range(len(image_dataset)):
        augFunc.K = image_dataset[files]
        augFunc.P = lab_dataset[files]
        # Separate here
        aug_str = rm.choice(augmentationList)()
        print("Counter: ", counter)
        counter = counter + 1
        print(aug_str)
        augmentation_save(files, aug_str)


def augmentation_ret(image_dataset, lab_dataset):
    augmentation_list_generation()
    for files in range(len(image_dataset)):
        augFunc.K = image_dataset[files]
        augFunc.P = lab_dataset[files]
        # Separate here
        aug_str = rm.choice(augmentationList)()
        print(aug_str)
        return (files, aug_str)
