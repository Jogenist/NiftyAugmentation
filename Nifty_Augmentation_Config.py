# ADDING A NEW METHOD:
# add the name of the new method in the Aug_Whtlst and in Augmentation_Config, then write the function in the Nifty_Augmentation-File under the functions block
# ----------------------------------------------------------------------------
# Set here to True or False if you want activate (1) or deactivate (0) a augmentation method.
#

Aug_Whtlst = {"rotate": 1,
              "scale": 1,
              "flip": 1,
              "translate": 1,
              "skew": 1,
              "blur": 1,
              "cropAndResize": 1,
              "cropAndPatch": 1,
              "elasticDistortion": 1,
              "randomErasing": 1,
              "noise": 1,
              "shear": 1,
              "saltAndPepper": 1,
              #.
              #.
              #.
              # add your new method here
              "noAugmentation": 1,
}

#Block Size
Block_Size = [64, 64, 64]

# ----------------------------------------------------------------------------
# Setting for Augmentation

rotate = {
    "angleMin": 0, #Degree
    "angleMax": 10, #Degree
}

scale = {
    "Min": 0.2,
    "Max": 0.5
}


flip = {
    "Min": 0.2,
    "Max": 0.5
}


translate = {
    "Min": 0.2,
    "Max": 0.5
}


skew = {
    "Min": 0.2,
    "Max": 0.5
}


blur = {
    "Min": 0.2,
    "Max": 0.5
}

cropAndResize = {
    "Min": 0.2,
    "Max": 0.5
}

cropAndPatch = {
    "Min": 0.2,
    "Max": 0.5
}


elasticDistortion = {
    "Min": 0.2,
    "Max": 0.5
}


randomErasing = {
    "Min": 0.2,
    "Max": 0.5
}


noise = {
    "Min": 0.2,
    "Max": 0.5
}

saltAndPepper = {
    "Min": 0.2,
    "Max": 0.5
}
# .
# .
# add your new method here
noAugmentation = {
    "Min": 0.2,
    "Max": 0.5
}






