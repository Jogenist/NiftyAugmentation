"""
Nifty_Augmentation.py

Contains functions and implementation for serveral augmentation methods. Is used in main.py to augment Nifty-Files.
Uses Nifty_Augmentation_Config for user-customizable augmentation parameters.

06/2021
"""
# ----------------------------------------------------------------------------
# import libraries
import random as rm
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from rotations import x_rotmat
from rotations import y_rotmat
from rotations import z_rotmat
import Nifty_Augmentation_Config as augConfig
from scipy.ndimage.interpolation import geometric_transform
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

# Global storage
K = 0
P = 0
# ----------------------------------------------------------------------------
# random axis rotations
rot_array = [x_rotmat, y_rotmat, z_rotmat]


def rotmat(matrix):  # this function is called when a rotation matrix over a random axis (x,y,z) is needed
    random_axis = rm.randrange(0, 3, 1)
    # if random_axis == 0:
    #     print("x_rotation")
    #
    # elif random_axis == 1:
    #     print("y_rotation")
    #
    # elif random_axis == 2:
    #     print("z_rotation")

    return rot_array[random_axis](matrix)


# -------------------------------------ROTATE-----------------------------------------------
def rotate():
    print("---rotate---")
    """
    Simple Image Rotation as compared in [O'Gara2019].
    [O'Gara2019] O'Gara, McGuinness, "Comparing Data Augmentation Strategies for Deep Image Classification", in
    IMVIP 2019 Irish Machine Vision and Image Procession, 2019.       
    """
    angle = rm.uniform(augConfig.rotate["angleMin"] * math.pi / 180,
                       augConfig.rotate["angleMax"] * math.pi / 180)  # randomly calculate rotation angle
    print(angle)
    M = rotmat(angle)  # calculate rotation matrix with given angle
    print("M: ", M)
    global K
    global P
    K = affine_transform(K, M, [0, 0, 0], output_shape=(64, 64, 64),
                         order=1)  # apply rotation matrix as an affine transformation
    P = affine_transform(P, M, [0, 0, 0], output_shape=(64, 64, 64),
                         order=1)  # apply same augmentation to the corresponding segmentation file
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "rotated"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------SCALE-----------------------------------------------
def scale():
    print("---scale---")
    """
    Scale as used in [Hussain].
    [Hussain] Hussain, Gimenez, Yi, Rubin, "Differential Data Augmentation Techniques for Medical Imaging "in
    Classification Tasks" from Stanford University, Department of Computer Science & Department of Radiology.       
    """
    scale = rm.uniform(augConfig.scale["Min"], augConfig.scale["Max"])  # randomly select scale factor
    M = rotmat(0) * scale  # calculate transformation matrix (no rotation) with given scale
    print("M: ", M)
    global K
    global P
    K = affine_transform(K, M, [0, 0, 0], output_shape=(64, 64, 64),
                         order=1)  # apply transformation as an affine transformation
    P = affine_transform(P, M, [0, 0, 0], output_shape=(64, 64, 64),
                         order=1)  # apply same augmentation to the corresponding segmentation file
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "scaled"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------FLIP-----------------------------------------------
def flip():
    print("---flip---")
    global K
    global P
    flipAxis = rm.randrange(0, 1, 1)  # randomly select flip axis
    K = np.flip(K, flipAxis)  # apply transformation as an affine transformation
    P = np.flip(P, flipAxis)  # apply same augmentation to the corresponding segmentation file
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20  # start at layer 10
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "flipped"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------TRANSLATE-----------------------------------------------
def translate():
    print("---translate---")
    translation = [rm.uniform(augConfig.translate["xMin"], augConfig.translate["xMax"]),
                   # randomly select translation parameters
                   rm.uniform(augConfig.translate["yMin"], augConfig.translate["yMax"]),
                   rm.uniform(augConfig.translate["zMin"], augConfig.translate["zMax"])]
    global K
    global P
    K = affine_transform(K, rotmat(0), translation, output_shape=(64, 64, 64),
                         order=1)  # apply translation as an affine transformation
    P = affine_transform(P, rotmat(0), translation, output_shape=(32, 32, 32),
                         order=1)  # apply same augmentation to the corresponding segmentation file
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "translated"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------SKEW-----------------------------------------------
def skew():
    print("---skew---")
    global K
    global P
    K = np.empty(augConfig.Block_Size)  # create empty numpy arrays with size of Nifty-File
    P = np.empty(augConfig.Block_Size)
    dl = rm.randrange(augConfig.skew["Min"], augConfig.skew["Max"])  # randomly select skew angle
    print("Skew Angle: ", dl)

    for n in range(K.shape[2]):  # go through each nifty slice
        h, l = K[:, :, n].shape  # get shape of 2D Image of this nifty slice

        def mapping(lc):  # complicated calculations
            l, c = lc
            dec = (dl * (l - h)) / h
            return l, c + dec

        def crop_center(img, cropx, cropy):
            y, x = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx]

        skew_k = geometric_transform(K[:, :, n], mapping, (h, l + dl), order=5,
                                     mode='nearest')  # apply skew augmentation as geometric transformation
        skew_p = geometric_transform(P[:, :, n], mapping, (h, l + dl), order=5,
                                     mode='nearest')  # apply same augmentation to the corresponding segmentation file
        skew_crop_k = crop_center(skew_k, 64, 64)  # crop skewed image to 64,64
        skew_crop_p = crop_center(skew_p, 64, 64)  # crop skewed label to 64,64
        K[:, :, n] = skew_crop_k  # append skewed slice to numpy array (image)
        P[:, :, n] = skew_crop_p  # append skewed slice to numpy array (segmenation)
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "skewed"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------BLUR-----------------------------------------------
def blur():
    print("---blur---")
    translation = [rm.uniform(augConfig.blur["Min"], augConfig.blur["Max"]), rm.uniform(augConfig.blur["Min"],
                                                                                        augConfig.blur["Max"]),
                   rm.uniform(augConfig.blur["Min"],
                              augConfig.blur["Max"])]  # randomly select translation parameters for bluring

    global K
    global P
    K = affine_transform(K, rotmat(0), translation, output_shape=(64, 64, 64),
                         order=1)  # apply blur as affine transformation
    P = affine_transform(P, rotmat(0), translation, output_shape=(64, 64, 64),
                         order=1)  # apply same augmentation to the corresponding segmentation file
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20  # start at layer 10
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "blurred"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------RANDOM CROP AND RESIZE-----------------------------------------------
def cropAndResize():
    print("---Crop & Resize---")
    global K
    global P
    K = np.empty(augConfig.Block_Size)  # create empty numpy arrays with size of Nifty-File
    P = np.empty(augConfig.Block_Size)
    a = rm.randrange(augConfig.cropAndResize["Min"],
                     augConfig.cropAndResize["Max"])  # randomly select starting point of crop
    print("a:", a)

    for n in range(K.shape[2]):  # go through each nifty slice
        crop_k = K[a:a + 16, a:a + 16, n]  # crop size is always 16x16
        crop_p = P[a:a + 16, a:a + 16, n]
        crop_resize_k = resize(crop_k, (64, 64))
        crop_resize_p = resize(crop_p, (64, 64))
        K[:, :, n] = crop_resize_k  # append skewed slice to numpy array (image)
        P[:, :, n] = crop_resize_p
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "crop_resize"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------CROP AND PATCH-----------------------------------------------
def cropAndPatch():
    print("---Crop & Patch---")
    """
    Image cropping and patching as proposed in [Takahashi2015].
    [Takahasi2015] Takahashi, Matsubara and Uehara, "Data Augmentation using Random Image Cropping and 
    Patching for Deep CNNs", in Journal of Latex Class Files, 2015.
    """
    global K
    global P
    K = np.empty(augConfig.Block_Size)  # create empty numpy arrays with size of Nifty-File
    P = np.empty(augConfig.Block_Size)
    for n in range(K.shape[2]):  # go through each nifty slice
        crop_k1 = K[0:32, 0:32, n]  # crop first slice
        crop_p1 = P[0:32, 0:32, n]
        crop_k2 = K[32:64, 0:32, n]  # crop second slice
        crop_p2 = P[32:64, 0:32, n]
        crop_k3 = K[32:64, 32:64, n]  # crop third slice
        crop_p3 = P[32:64, 32:64, n]
        crop_k4 = K[0:32, 32:64, n]  # crop fourth slice
        crop_p4 = P[0:32, 32:64, n]
        # rebuild k
        crop_ka = np.concatenate((crop_k3, crop_k2), axis=0)
        crop_kb = np.concatenate((crop_k4, crop_k1), axis=0)
        crop_k = np.concatenate((crop_kb, crop_ka), axis=1)
        # rebuild p
        crop_pa = np.concatenate((crop_p3, crop_p2), axis=0)
        crop_pb = np.concatenate((crop_p4, crop_p1), axis=0)
        crop_p = np.concatenate((crop_pb, crop_pa), axis=1)
        K[:, :, n] = crop_k  # append patched image to numpy array
        P[:, :, n] = crop_p  # append patched label to numpy array
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20  # start at layer 10
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "crop_patch"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------ELASTIC DISTORTION-----------------------------------------------
def elasticDistortion():
    print("---Elastic Distortion---")
    """
    Elastic deformation as described in [Simard2003].
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    global K
    global P
    K = np.empty(augConfig.Block_Size)  # create empty numpy arrays with size of Nifty-File
    P = np.empty(augConfig.Block_Size)
    alpha = augConfig.elasticDistortion["alpha"]  # load alpha and sigma parameters from ConfigFile
    sigma = augConfig.elasticDistortion["sigma"]
    print("alpha:", alpha)
    print("sigma: ", sigma)
    random_state = np.random.RandomState(None)
    for n in range(K.shape[2]):  # go through each nifty slice
        shape = K[n].shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        K[:, :, n] = map_coordinates(K[n], indices, order=1).reshape(
            shape)  # append distorted slice to numpy array (image)
        P[:, :, n] = map_coordinates(P[n], indices, order=1).reshape(
            shape)  # append distorted slice to numpy array (label)
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "elastic_distortion"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------RANDOM ERASING-----------------------------------------------
def randomErasing2():
    print("---Random Erasing---")
    """
    Random Erasing as compared in [O'Gara2019].
    [O'Gara2019] O'Gara, McGuinness, "Comparing Data Augmentation Strategies for Deep Image Classification", in
    IMVIP 2019 Irish Machine Vision and Image Procession, 2019.
    """
    i = 20  # start at layer 10
    A = np.empty(augConfig.Block_Size)
    B = np.empty(augConfig.Block_Size)
    a = rm.randrange(0, 48)  # randomly select pixels to be erased
    b = 10  # set size of erased area here
    for n in range(K.shape[2]):  # go through each nifty slice
        K[a:a + b, a:a + b, n] = 0
        P[a:a + b, a:a + b, n] = 0
        A[:, :, n] = K[:, :, n]
        B[:, :, n] = P[:, :, n]
    print(A.shape)
    plt.imshow(A[:, :, i])
    plt.show()
    i = i + 1
    aug_str = "random_erasing"

def randomErasing():
    print("---Random Erasing---")
    """
    Random Erasing as compared in [O'Gara2019].
    [O'Gara2019] O'Gara, McGuinness, "Comparing Data Augmentation Strategies for Deep Image Classification", in
    IMVIP 2019 Irish Machine Vision and Image Procession, 2019.
    """
    global K
    global P
    K_intern = np.empty(augConfig.Block_Size)  # create empty numpy arrays with size of Nifty-File
    P_intern = np.empty(augConfig.Block_Size)
    a = rm.randrange(augConfig.randomErasing["Min"], augConfig.randomErasing["Max"])
    # randomly select pixels to be erased
    b = 10
    # set size of erased area here
    for n in range(K.shape[2]):
        # go through each nifty slice
        K[a:a + b, a:a + b, n] = 0
        # delete values at given location in image (set to 0 -> black)
        P[a:a + b, a:a + b, n] = 0
        K_intern[:, :, n] = K[:, :, n]
        P_intern[:, :, n] = P[:, :, n]
    print(K_intern.shape)
    if augConfig.PlotMode:
        # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K_intern[:, i])
        plt.show()
    aug_str = "random_erasing"
    # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------NOISE-----------------------------------------------
def noise():
    print("---noise---")
    global K
    global P
    print(K)
    noise = np.random.normal(augConfig.noise["Min"], augConfig.noise["Max"],
                             K.shape)  # create numpy array filled with random values
    K = K + noise  # add noise numpy array to Nifty numpy array
    P = P + noise
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "noised"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------SCALE-----------------------------------------------
def shear():
    print("---shear---")
    """
    Shear as used in [Hussain].
    [Hussain] Hussain, Gimenez, Yi, Rubin, "Differential Data Augmentation Techniques for Medical Imaging "in
    Classification Tasks" from Stanford University, Department of Computer Science & Department of Radiology.       
    """
    shear = rm.uniform(augConfig.shear["Min"], augConfig.shear["Max"])  # randomly select shear factor
    M = rotmat(0)  # create rotation matrix with angle = 0
    M[1][0] = shear  # apply shear factor to first element of rotation matrix
    print("M: ", M)
    print("M2: ", M[0][1])
    global K
    global P
    K = affine_transform(K, M, [0, 0, 0], output_shape=(64, 64, 64),
                         order=1)  # apply given rotation matrix to nifty-file with affine transformation
    P = affine_transform(P, M, [0, 0, 0], output_shape=(64, 64, 64),
                         order=1)  # apply same augmentation to the corresponding segmentation file
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "sheared"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------SALT & PEPPER-----------------------------------------------
def saltAndPepper():
    print("---Salt and Pepper---")
    global K
    global P
    print(K)
    i = 20  # start at layer 10
    K_intern = np.empty(augConfig.Block_Size)  # create empty numpy arrays with size of Nifty-File
    P_intern = np.empty(augConfig.Block_Size)
    a = np.random.randint(0, 64, augConfig.saltAndPepper["amount"])  # randomly select pixels for pepper
    b = np.random.randint(0, 64, augConfig.saltAndPepper["amount"])
    c = np.random.randint(0, 64, augConfig.saltAndPepper["amount"])  # randomly select pixels for salt
    d = np.random.randint(0, 64, augConfig.saltAndPepper["amount"])
    for n in range(K.shape[2]):  # go through each nifty slice
        for p in range(len(a)):
            K[a[p], b[p], n] = 1  # apply pepper
            P[a[p], b[p], n] = 1
            K[c[p], d[p], n] = 0  # apply salt
            P[c[p], d[p], n] = 0
            K_intern[:, :, n] = K[:, :, n]
            P_intern[:, :, n] = P[:, :, n]
    print(K_intern.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot augmented image
        i = 20
        plt.imshow(K_intern[:, i])
        plt.show()
    K = K_intern
    P = P_intern
    aug_str = "salted&peppered"  # define augmentation string used for naming the augmented files
    return aug_str


# -------------------------------------NO AUGMENTATION-----------------------------------------------
# TODO: not working because the file is not written to K
def noAugmentation():
    print("---no augmentation---")
    global K
    global P
    K = K  # save unaugmented file as new numpy array
    P = P
    print(K.shape)
    if augConfig.PlotMode:  # if PlotMode is On, plot unaugmented image
        i = 20
        plt.imshow(K[:, i])
        plt.show()
    aug_str = "noAugmentation"  # define augmentation string used for naming the augmented files
    return aug_str  # dont save unaugmented image, therefore leave here
