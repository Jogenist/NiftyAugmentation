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
# Select augmentation here:
Augmentation_Config ={
    "rotate": {
        "active": 0
    },
    "scale": {
        "active": 1
    },
    "flip":{
       "active": 1
    },
    "translate":{
        "active": 1
    },
    "skew": {
        "active": 1
    },
    "blur": {
        "active": 1
    },
    "cropAndResize": {
        "active": 1
    },
    "cropAndPatch": {
        "active": 1
    },
    "elasticDistortion": {
        "active": 1
    },
    "randomErasing": {
        "active": 1
    },
    "noise": {
        "active": 1
    },
    "saltAndPepper": {
        "active": 1
    },
    # .
    # .
    # .
    # add your new method here
    "no augmentation": {
        "active": 1
    }
}

# augmentationList = []
#
#
# def augmentation_list():
#     global augmentationList
#     for x in Aug_Whtlst:
#         if Aug_Whtlst[x]:
#             augmentationList.append(x)
#
# augmentation_list()
# print(augmentationList)


