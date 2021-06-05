"""
Nifty_Augmentation_Config.py

Parameters and Configurations for the augmentation functions are defined and configured here.
This file is used by Nifty_Augmentation.py to apply the augmentations with the user-customizable parameters.

06/2021
"""

# ADDING A NEW METHOD:
# add the name of the new method in the Aug_Whtlst and in Augmentation_Config,
# then write the function in the Nifty_Augmentation-File under the functions block
# ----------------------------------------------------------------------------
# Set here to True or False if you want activate (1) or deactivate (0) a augmentation method.
#

PlotMode = 1
"""
0: Off -> no Plots
1: On -> Plot each augmentation
"""


Aug_Whtlst = {"rotate":             1,
              "scale":              1,
              "flip":               1,
              "translate":          1,
              "skew":               1,
              "blur":               1,
              "cropAndResize":      1,
              "cropAndPatch":       1,
              "elasticDistortion":  1,
              "randomErasing":      1,
              "noise":              1,
              "shear":              1,
              "saltAndPepper":      1,
              # .
              # .
              # .
              # add your new method here
              "noAugmentation":     1
}

# Block Size
Block_Size = [64, 64, 64]

# ----------------------------------------------------------------------------
# Setting for Augmentation

rotate = {
    """
    Minimum and maximum angle for rotation augmentation.
    Rotation angle is randomly selected in this range.
    """
    
    "angleMin": 0,  # Degree
    "angleMax": 10  # Degree
}

scale = {
    """
    Minimum and maximum factor for scaling.
    Scale factor is randomly selected in this range.
    """
    
    "Min": 0.2,
    "Max": 0.5
}


flip = {
    """
    No parameters needed for flip augmentation.
    """
}


translate = {
    """
    Minimum and maximum values for translation in x,y and z direction.
    Translation values are randomly selected in the given range.
    """
    
    "xMin": -30,
    "xMax": 30,
    "yMin": -30,
    "yMax": 30,
    "zMin": -30,
    "zMax": 30
}


skew = {
    """
    Minimum and maximum skew angle for skew augmentation.
    Skew angle is randomly selected in this range.
    """
    
    "Min": 5,
    "Max": 40
}


blur = {
    """
    Minimum and maximum translation values for blurring.
    Translation value for blurring is selected in this range.
    """
    
    "Min": 0,
    "Max": 0.8
}

cropAndResize = {
    """
    Minimum and maximum value for start point of cropping.
    Cropping start point is selected in this range.
    """
    
    "Min": 0,
    "Max": 48
}

cropAndPatch = {
    """
    No parameters needed for Crop&Patch augmentation.
    """
}


elasticDistortion = {
    """
    Alpha and Sigma parameters for elastic distortion augmentation.
    Use alpha = 20 and sigma = 1 to create art.
    """
    
    "alpha": 55,
    "sigma": 10
}


randomErasing = {
    """
    Minimum and maximum value for selecting pixels to be erased.
    Pixels are selected in this range.
    Minimum and maximum size of erased area.
    """
    
    "Min": 0,
    "Max": 48,
    "sizeMin": 2,
    "sizeMax": 15
}


noise = {
    """
    Minimum and maximum value for noise values added to the nifty-file.
    Values are selected randomly in this range.
    """
    
    "Min": 0,
    "Max": 12
}

shear = {
    """
    Minimum and maximum value for shear factor.
    Shear factor is selected in this range.
    """
    
    "Min": 0.4,
    "Max": 1

}

saltAndPepper = {
    """
    Amount of Salt and Pepper grains added to the nifty-files.
    """
    
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






