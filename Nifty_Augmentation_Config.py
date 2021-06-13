"""
Nifty_Augmentation_Config.py

Parameters and Configurations for the augmentation functions are defined and configured here.
This file is used by Nifty_Augmentation.py to apply the augmentations with the user-customizable parameters.

06/2021
"""
# ----------------------------------------------------------------------------
PlotMode = 1
"""
0: Off -> no Plots
1: On -> Plot each augmentation
"""

# ----------------------------------------------------------------------------
# ADDING A NEW METHOD:
"""
 add the name of the new method in the Aug_Whtlst and in Augmentation_Config,
 then write the function in the Nifty_Augmentation-File under the functions block
 Set here to True or False if you want activate (1) or deactivate (0) a augmentation method.
"""
Aug_Whtlst = {
    "rotate": 1,
    "scale": 1,
    "flip": 1,
    "translate": 0,
    "skew": 0,
    "blur": 0,
    "cropAndResize": 0,
    "cropAndPatch": 0,
    "elasticDistortion": 0,
    "randomErasing": 0,
    "noise": 0,
    "shear": 0,
    "saltAndPepper": 0,
    # .
    # .
    # .
    # add your new method here
    "noAugmentation": 0
}
# ----------------------------------------------------------------------------
# Block Size
"""
change the block size here, to the resolution of your nii-file
"""
Block_Size = [64, 64, 64]

# ----------------------------------------------------------------------------
# Priority Setting of each Mode
"""
set the priority of SINGLE, MULIPLE_ORDERED, MULIPLE_RANDOM to a number how
often you want to execute this mode, compared to the other mode. 
Example:
    1. in Whitelist only rotation an scale are allowed.
    2. Ordered funtions are: [rotate, scale]
    3. random deepth is 3 (this makes no sense, because with only 2 functions
                           and a ordered augmentation, you also allow unordered.
                           But its only for example)
    3.  Single_Mode_Priority = 1
        Multiple_Mode_Ordered_Priority = 1
        Multiple_Mode_Unordered_Priority = 1
    --> this leads to a execution of:
            25 % rotate, 25 % scale, 25 % multiple, 25 % random
    
    4. If the priority of random is changed to 2, then we get:
        Single_Mode_Priority = 1
        Multiple_Mode_Ordered_Priority = 1
        Multiple_Mode_Unordered_Priority = 2
            20 % rotate, 20 % scale, 20 % multiple, 40 % random
    5. You can also deactivate one of the executions here, if you set the 
        priority to 0.
"""
Single_Mode_Priority = 1
Multiple_Mode_Ordered_Priority = 1
Multiple_Mode_Unordered_Priority = 1

# ----------------------------------------------------------------------------
# Multiple Augmentation Mode ordered
"""For a bigger data-package you can easily do multiple Augmentations. But in most of the cases the
final augmented product if to far from reality, that's why you have here the options to choose your own order of
multiple augmentations.
 """
# Set the following mode to activate the augmentation:
# False = Deactivated
# True = ordered Augmentations from ordered Whitelist below
Multiple_Mode_ordered_Active = True

# Ordered multiple Augmentation
Aug_Whtlst_Multiple_ordered = [
    ["rotate", "scale", "flip"],
    ["rotate", "skew", "cropAndResize"]
]

# Multiple Augmentations unordered:
# 
"""
For crazy augmentations without any 
!!!!!!!! This can cause issues and the resolution can be something stange !!!!!!!!!!!!

"""
# False = Deactivated
# True = unordered Augmentations for Whitelist
Multiple_Mode_unordered_Active = True
# Set Multiple_Aug_Deep to the number how many functions should be called behind each other
# deactivated function from whitelist are not called
Multiple_Aug_Depth_min = 1
Multiple_Aug_Depth_max = 3





# ----------------------------------------------------------------------------
# Setting for Augmentation

# ----ROTATE ----
"""
Minimum and maximum angle for rotation augmentation.
Rotation angle is randomly selected in this range.
"""
rotate = {

    "angleMin": 0,  # Degree
    "angleMax": 10  # Degree
}

# ---- SCALE ----
"""
Minimum and maximum factor for scaling.
Scale factor is randomly selected in this range.
"""
scale = {
    "Min": 0.2,
    "Max": 0.5
}

# ---- FLIP ----
flip = {
    """
    No parameters needed for flip augmentation.
    """
}

# ---- TRANSLATE ----
"""
Minimum and maximum values for translation in x,y and z direction.
Translation values are randomly selected in the given range.
"""
translate = {
    "xMin": -30,
    "xMax": 30,
    "yMin": -30,
    "yMax": 30,
    "zMin": -30,
    "zMax": 30
}

# ---- SKEW ----
"""
Minimum and maximum skew angle for skew augmentation.
Skew angle is randomly selected in this range.
"""
skew = {
    "Min": 5,
    "Max": 40
}

# ---- BLUR ----
"""
Minimum and maximum translation values for blurring.
Translation value for blurring is selected in this range.
"""
blur = {
    "Min": 0,
    "Max": 0.8
}

# ---- CROP AND RESIZE ----
"""
Minimum and maximum value for start point of cropping.
Cropping start point is selected in this range.
"""
cropAndResize = {
    "Min": 0,
    "Max": 48
}

# --- CROP AND PATCH
"""
No parameters needed for Crop&Patch augmentation.
"""
cropAndPatch = {

}

# ---- ELASTIC DISTORTION ----
"""
Alpha and Sigma parameters for elastic distortion augmentation.
Use alpha = 20 and sigma = 1 to create art.
"""
elasticDistortion = {
    "alpha": 55,
    "sigma": 10
}

# ---- RANDOM ERASE ----
"""
Minimum and maximum value for selecting pixels to be erased.
Pixels are selected in this range.
Minimum and maximum size of erased area.
"""
randomErasing = {
    "Min": 0,
    "Max": 48,
    "sizeMin": 2,
    "sizeMax": 15
}

# -----NOISE----
"""
Minimum and maximum value for noise values added to the nifty-file.
Values are selected randomly in this range.
"""
noise = {
    "Min": 0,
    "Max": 12
}

# ----SHEAR----
"""
Minimum and maximum value for shear factor.
Shear factor is selected in this range.
"""
shear = {
    "Min": 0.4,
    "Max": 1

}

# ----SALT AND PEPPER----
"""
Amount of Salt and Pepper grains added to the nifty-files.
"""
saltAndPepper = {
    "amount": 13
}

# .
# .
# add your new method here

noAugmentation = {
    """
    No parameters needed when not augmenting.
    """
}
