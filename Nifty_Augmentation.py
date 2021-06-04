import random as rm
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
from rotations import x_rotmat  # from rotations.py
from rotations import y_rotmat  # from rotations.py
from rotations import z_rotmat  # from rotations.py
import Nifty_Augmentation_Config as augConfig
from scipy.ndimage.interpolation import geometric_transform
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


# ----------------------------------------------------------------------------
# random axis roations
# this function is called when a random axis rotation should be done
rot_array = [x_rotmat, y_rotmat, z_rotmat]


def rotmat(matrix):
    random_axis = rm.randrange(0, 3, 1)
    print("Ration: ", random_axis)
    if random_axis == 0:
        print("x_rotation")

    elif random_axis == 1:
        print("y_rotation")

    elif (random_axis) == 2:
        print("z_rotation")

    return rot_array[random_axis](matrix)

# -------------------------------------ROTATE-----------------------------------------------
def rotate(Img_dataset,Lab_dataset, file_int):
    print("---rotate---")
    """
    Simple Image Rotation as compared in [O'Gara2019].
    [O'Gara2019] O'Gara, McGuinness, "Comparing Data Augmentation Strategies for Deep Image Classification", in
    IMVIP 2019 Irish Machine Vision and Image Procession, 2019.       
    """
    angle = rm.uniform(augConfig.rotate["angleMin"]*math.pi/180, augConfig.rotate["angleMax"]*math.pi/180)
    print(angle)
    i = 20  # start at layer 10
    # rotation matrix for rotation of 0.2 radians around x axis
    M = rotmat(angle)
    print("M: ", M)
    translation = [0, 0, 0]  # Translation from I to J [y,z,x]
    global K
    global P
    # order=1 for linear interpolation
    K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "rotated"
    return file_int, aug_str
    # -------------------------------------SCALE-----------------------------------------------


def scale(Img_dataset,Lab_dataset, file_int):
    print("---scale---")
    """
    Scale as used in [Hussain].
    [Hussain] Hussain, Gimenez, Yi, Rubin, "Differential Data Augmentation Techniques for Medical Imaging "in
    Classification Tasks" from Stanford University, Department of Computer Science & Department of Radiology.       
    """
    i = 20  # start at layer 10
    scale = rm.uniform(augConfig.scale["Min"], augConfig.scale["Max"])  # random factor
    M = rotmat(0) * scale
    print("M: ", M)
    translation = [0, 0, 0]  # Translation from I to J [y,z,x]
    global K
    global P
    # order=1 for linear interpolation
    K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "scaled"
    return file_int, aug_str
# -------------------------------------FLIP-----------------------------------------------
def flip(Img_dataset,Lab_dataset, file_int):
    print("---flip---")
    i = 20  # start at layer 10
    global K
    global P
    flipAxis = rm.randrange(0,1,1)
    K = np.flip(Img_dataset[file_int], flipAxis)
    P = np.flip(Lab_dataset[file_int], flipAxis)
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "flipped"
    return file_int, aug_str
# -------------------------------------TRANSLATE-----------------------------------------------
def translate(Img_dataset,Lab_dataset, file_int):
    print("---translate---")
    i = 20  # start at layer 10
    translation = [-32.2, 24, -32.2]  # Translation from I to J [y,z,x]
    M = rotmat(0)
    # order=1 for linear interpolation
    global K
    global P
    K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(32, 32, 32), order=1)
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "translated"
    return file_int, aug_str

# -------------------------------------SKEW-----------------------------------------------
def skew(Img_dataset,Lab_dataset, file_int):
    print("---skew---")
    i = 20  # start at layer 10
    global K
    global P
    K = np.empty(augConfig.Block_Size)
    P = np.empty(augConfig.Block_Size)
    dl = rm.randrange(5, 40)  # skew angle
    print("Skew Angle: ", dl)

    for n in range(Img_dataset[file_int].shape[2]):  # go through each nifty slice
        h, l = Img_dataset[file_int][:, :, n].shape  # get shape of 2D Image of this nifty slice

        def mapping(lc):  # komplizierter shit
            l, c = lc
            dec = (dl * (l - h)) / h
            return l, c + dec

        def crop_center(img, cropx, cropy):
            y, x = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            return img[starty:starty + cropy, startx:startx + cropx]

        skew_k = geometric_transform(Img_dataset[file_int][:, :, n], mapping, (h, l + dl), order=5,
                                     mode='nearest')  # Skew the current slice
        skew_p = geometric_transform(Lab_dataset[file_int][:, :, n], mapping, (h, l + dl), order=5, mode='nearest')
        skew_crop_k = crop_center(skew_k, 64, 64)  # crop skewed image to 64,64
        skew_crop_p = crop_center(skew_p, 64, 64)  # crop skewed label to 64,64
        K[:, :, n] = skew_crop_k  # append skewed slice to numpy array (image)
        P[:, :, n] = skew_crop_p  # append skewed slice to numpy array (segmenation)
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "skewed"
    return file_int, aug_str

# -------------------------------------BLUR-----------------------------------------------
def blur(Img_dataset, Lab_dataset, file_int):
    print("---blur---")
    i = 20  # start at layer 10
    translation = [-0.5, 0, 0.5]  # Translation from I to J [y,z,x]
    M = rotmat(0)
    # order=1 for linear interpolation
    global K
    global P
    K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "blured"
    return file_int, aug_str

# -------------------------------------RANDOM CROP AND RESIZE-----------------------------------------------
def cropAndResize(Img_dataset, Lab_dataset, file_int):
    print("---Crop & Resize---")
    i = 20  # start at layer 10
    global K
    global P
    K = np.empty(augConfig.Block_Size)
    P = np.empty(augConfig.Block_Size)
    a = rm.randrange(0, 48)  # randomly select starting point of crop
    print("a:", a)

    for n in range(Img_dataset[file_int].shape[2]):  # go through each nifty slice
        crop_k = Img_dataset[file_int][a:a + 16, a:a + 16, n]  # crop size is always 16x16
        crop_p = Img_dataset[file_int][a:a + 16, a:a + 16, n]
        crop_resize_k = resize(crop_k, (64, 64))
        crop_resize_p = resize(crop_p, (64, 64))
        K[:, :, n] = crop_resize_k  # append skewed slice to numpy array (image)
        P[:, :, n] = crop_resize_p
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "crop_resize"
    return file_int, aug_str

# -------------------------------------CROP AND PATCH-----------------------------------------------
def cropAndPatch(Img_dataset, Lab_dataset, file_int):
    print("---Crop & Patch---")
    """
    Image cropping and patching as proposed in [Takahashi2015].
    [Takahasi2015] Takahashi, Matsubara and Uehara, "Data Augmentation using Random Image Cropping and 
    Patching for Deep CNNs", in Journal of Latex Class Files, 2015.
    """
    i = 20  # start at layer 10
    global K
    global P
    K = np.empty(augConfig.Block_Size)
    P = np.empty(augConfig.Block_Size)
    for n in range(Img_dataset[file_int].shape[2]):  # go through each nifty slice
        crop_k1 = Img_dataset[file_int][0:32, 0:32, n]  # crop first slice
        crop_p1 = Img_dataset[file_int][0:32, 0:32, n]
        crop_k2 = Img_dataset[file_int][32:64, 0:32, n]  # crop second slice
        crop_p2 = Img_dataset[file_int][32:64, 0:32, n]
        crop_k3 = Img_dataset[file_int][32:64, 32:64, n]  # crop third slice
        crop_p3 = Img_dataset[file_int][32:64, 32:64, n]
        crop_k4 = Img_dataset[file_int][0:32, 32:64, n]  # crop fourth slice
        crop_p4 = Img_dataset[file_int][0:32, 32:64, n]
        # rebuild k
        crop_ka = np.concatenate((crop_k3, crop_k2), axis=0)
        crop_kb = np.concatenate((crop_k4, crop_k1), axis=0)
        crop_k = np.concatenate((crop_kb, crop_ka), axis=1)
        # rebuild p
        crop_pa = np.concatenate((crop_p3, crop_p2), axis=0)
        crop_pb = np.concatenate((crop_p4, crop_p1), axis=0)
        crop_p = np.concatenate((crop_pb, crop_pa), axis=1)
        K[:, :, n] = crop_k  # append skewed slice to numpy array (image)
        P[:, :, n] = crop_p
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "crop_patch"
    return file_int, aug_str

# -------------------------------------ELASTIC DISTORTION-----------------------------------------------
def elasticDistortion(Img_dataset, Lab_dataset, file_int):
    print("---Elastic Distortion---")
    """
    Elastic deformation as described in [Simard2003].
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """
    i = 20  # start at layer 10
    global K
    global P
    K = np.empty(augConfig.Block_Size)
    P = np.empty(augConfig.Block_Size)
    # ----- use this settings to create art ------
    # alpha = 20
    # sigma = 1
    # --------------------------------------------
    alpha = 55
    sigma = 10
    # alpha = rm.randrange(1, 10)  # randomly select starting point of crop
    print("alpha:", alpha)
    # sigma = rm.randrange(1,10)
    print("sigma: ", sigma)
    random_state = np.random.RandomState(None)
    for n in range(Img_dataset[file_int].shape[2]):  # go through each nifty slice
        shape = Img_dataset[file_int][n].shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        K[:, :, n] = map_coordinates(Img_dataset[file_int][n], indices, order=1).reshape(
            shape)  # append distorted slice to numpy array (image)
        P[:, :, n] = map_coordinates(Lab_dataset[file_int][n], indices, order=1).reshape(
            shape)  # append distorted slice to numpy array (label)
    print(K.shape)
    plt.imshow(K[:, :, i])
    plt.show()
    i = i + 1
    aug_str = "elastic_distortion"
    return file_int, aug_str

# -------------------------------------RANDOM ERASING-----------------------------------------------
def randomErasing(Img_dataset, Lab_dataset, file_int):
    print("---Random Erasing---")
    """
    Random Erasing as compared in [O'Gara2019].
    [O'Gara2019] O'Gara, McGuinness, "Comparing Data Augmentation Strategies for Deep Image Classification", in
    IMVIP 2019 Irish Machine Vision and Image Procession, 2019.
    """
    i = 20  # start at layer 10
    global K
    global P
    K = np.empty(augConfig.Block_Size)
    P = np.empty(augConfig.Block_Size)
    a = rm.randrange(0, 48)  # randomly select pixels to be erased
    b = 10  # set size of erased area here
    for n in range(Img_dataset[file_int].shape[2]):  # go through each nifty slice
        Img_dataset[file_int][a:a + b, a:a + b, n] = 0
        Lab_dataset[file_int][a:a + b, a:a + b, n] = 0
        K[:, :, n] = Img_dataset[file_int][:, :, n]
        P[:, :, n] = Lab_dataset[file_int][:, :, n]
    print(K.shape)
    plt.imshow(K[:, :, i])
    plt.show()
    i = i + 1
    aug_str = "random_erasing"
    return file_int, aug_str

# -------------------------------------NOISE-----------------------------------------------
def noise(Img_dataset, Lab_dataset, file_int):
    print("---noise---")
    i = 20  # start at layer 10
    print(Img_dataset[file_int])
    noise = np.random.normal(0, 12, Img_dataset[file_int].shape)
    global K
    global P
    K = Img_dataset[file_int] + noise
    P = Lab_dataset[file_int] + noise
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "noised"
    return file_int, aug_str

# -------------------------------------SCALE-----------------------------------------------
def shear(Img_dataset, Lab_dataset, file_int):
    print("---shear---")
    """
    Shear as used in [Hussain].
    [Hussain] Hussain, Gimenez, Yi, Rubin, "Differential Data Augmentation Techniques for Medical Imaging "in
    Classification Tasks" from Stanford University, Department of Computer Science & Department of Radiology.       
    """
    i = 20  # start at layer 10
    shear = rm.uniform(0.4, 1)  # random factor
    M = rotmat(0)
    # M[0][1] = 1
    M[1][0] = shear
    print("M: ", M)
    print("M2: ", M[0][1])
    translation = [0, 0, 0]  # Translation from I to J [y,z,x]
    # order=1 for linear interpolation
    global K
    global P
    K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    aug_str = "sheared"
    return file_int, aug_str

# -------------------------------------SALT & PEPPER-----------------------------------------------
def saltAndPepper(Img_dataset, Lab_dataset, file_int):
    print("---Salt and Pepper---")
    print(Img_dataset[file_int])
    i = 20  # start at layer 10
    global K
    global P
    K = np.empty(augConfig.Block_Size)
    P = np.empty(augConfig.Block_Size)
    a = np.random.randint(0, 64, 13)  # randomly select pixels for pepper
    b = np.random.randint(0, 64, 13)
    c = np.random.randint(0, 64, 13)  # randomly select pixels for salt
    d = np.random.randint(0, 64, 13)
    for n in range(Img_dataset[file_int].shape[2]):  # go through each nifty slice
        for p in range(len(a)):
            Img_dataset[file_int][a[p], b[p], n] = 1  # apply pepper
            Lab_dataset[file_int][a[p], b[p], n] = 1
            Img_dataset[file_int][c[p], d[p], n] = 0  # apply salt
            Lab_dataset[file_int][c[p], d[p], n] = 0
            K[:, :, n] = Img_dataset[file_int][:, :, n]
            P[:, :, n] = Lab_dataset[file_int][:, :, n]
    print(K.shape)
    plt.imshow(K[:, :, i])
    plt.show()
    i = i + 1
    aug_str = "salted&peppered"
    return file_int, aug_str

# -------------------------------------NO AUGMENTATION-----------------------------------------------
def noAugmentation(Img_dataset, Lab_dataset, file_int):
    print("---no augmentation---")
    i = 20  # start at layer 10
    global K
    global P
    K = Img_dataset[file_int]
    P = Lab_dataset[file_int]
    print(K.shape)
    plt.imshow(K[:, i])
    plt.show()
    i = i + 1
    return ()  # dont save unaugmented image, therefore leave here
