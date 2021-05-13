import numpy as np
np.set_printoptions(precision=4)  # print arrays to 4DP (float point precision, round to 4 digits)
import matplotlib.pyplot as plt
# gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
from scipy.ndimage import affine_transform
from scipy.ndimage.interpolation import geometric_transform
from scipy.ndimage import shift
from rotations import x_rotmat  # from rotations.py
from rotations import y_rotmat  # from rotations.py
from rotations import z_rotmat  # from rotations.py
from skimage.transform import resize
import nibabel as nib
import os
import random as rm


# ----------------------------------------------------------------------------
DATADIR = "Sample Data"  # Data directory
Block_Size = [64, 64, 64]
Img_dataset = []
Lab_dataset = []
Img_str_lst = []  # Image Structure List?
Lab_str_lst = []  # Laboratory Structure List?

# ----------------------------------------------------------------------------
# Select augmentation here:

Aug_Whtlst = ["rotate", "scale", "flip", "translate", "skew", "blur", "no augmentation"]

# ----------------------------------------------------------------------------
# load all nii image files from datadir in dataset
os.chdir(DATADIR + "/Image")  # change the current working directory to DATADIR
counter = 0
for nii in os.listdir(os.getcwd()):
    Img_str_lst.append(nii[:-4])  # append tha images to a list without the file ending ( [:-4] removes the .nii)
    img = nib.load(nii)  # load nifty file
    nii_data = img.get_fdata()  # get the data as an float array
    # print(nii_data)
    I = nii_data[..., 0]  # I is the first volume
    # print(I)
    Img_dataset.append(I)
Img_dataset = np.asarray(Img_dataset)

# load all nii segmentation files from datadir in dataset
os.chdir("..")  # go back one directory
os.chdir("Label")
for nii in os.listdir(os.getcwd()):
    Lab_str_lst.append(nii[:-4])
    lab = nib.load(nii)
    nii_data = lab.get_fdata()
    I = nii_data[..., 0]  # I is the first volume
    Lab_dataset.append(I)
    # print(Lab_dataset.shape)
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
# random axis roations
# this function is called when a random axis roatation should be done
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


# ----------------------------------------------------------------------------

# augment given nii file
def augmentation(aug_int, file_int):
    # -------------------------------------ROTATE-----------------------------------------------
    if aug_int == 0:
        print("---rotate---")
        angle = rm.uniform(0.1, 0.5)
        print(angle)
        i = 10  # start at layer 10
        # rotation matrix for rotation of 0.2 radians around x axis
        M = rotmat(angle)
        translation = [0, 0, 0]  # Translation from I to J [y,z,x]
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "rotated"

    # -------------------------------------SCALE-----------------------------------------------
    if aug_int == 1:
        print("---scale---")
        i = 10  # start at layer 10
        # rotation matrix for rotation of 0.2 radians around x axis
        M = rotmat(0)
        translation = [0, 0, 0]  # Translation from I to J [y,z,x]
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(32, 32, 32), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(32, 32, 32), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "scaled"
    # -------------------------------------FLIP-----------------------------------------------
    if aug_int == 2:
        print("---flip---")
        i = 10  # start at layer 10
        K = np.flip(Img_dataset[file_int], 1)
        P = np.flip(Lab_dataset[file_int], 1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "flipped"
    # -------------------------------------TRANSLATE-----------------------------------------------
    if aug_int == 3:
        print("---translate---")
        i = 10  # start at layer 10
        translation = [-32.2, 24, -32.2]  # Translation from I to J [y,z,x]
        M = y_rotmat(0)
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(32, 32, 32), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "translated"
    # -------------------------------------SKEW-----------------------------------------------
    if aug_int == 4:
        print("---skew---")
        i = 10  # start at layer 10
        K = np.empty(Block_Size)
        P = np.empty(Block_Size)
        dl = rm.randrange(5, 40)  # skew angle
        print("Skew Angle: ", dl)
        for n in range(Img_dataset[file_int].shape[2]):         # go through each nifty slice
            h, l = Img_dataset[file_int][:,:,n].shape           # get shape of 2D Image of this nifty slice

            def mapping(lc):                                    # komplizierter shit
                l, c = lc
                dec = (dl * (l - h)) / h
                return l, c + dec

            def crop_center(img, cropx, cropy):
                y, x = img.shape
                startx = x // 2 - (cropx // 2)
                starty = y // 2 - (cropy // 2)
                return img[starty:starty + cropy, startx:startx + cropx]

            skew_k = geometric_transform(Img_dataset[file_int][:, :, n], mapping, (h, l + dl), order=5, mode='nearest') # Skew the current slice
            skew_p = geometric_transform(Lab_dataset[file_int][:, :, n], mapping, (h, l + dl), order=5, mode='nearest')
            skew_crop_k = crop_center(skew_k,64, 64)            # crop skewed image to 64,64
            skew_crop_p = crop_center(skew_p, 64, 64)           # crop skewed label to 64,64
            K[:, :, n] = skew_crop_k                            # append skewed slice to numpy array (image)
            P[:, :, n] = skew_crop_p                            # append skewed slice to numpy array (segmenation)
        i = i + 1
        aug_str = "skewed"
    # -------------------------------------BLUR-----------------------------------------------
    if aug_int == 5:
        print("---blur---")
        i = 10  # start at layer 10
        translation = [-0.5, 0, 0.5]  # Translation from I to J [y,z,x]
        M = y_rotmat(0)
        # order=1 for linear interpolation
        K = affine_transform(Img_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        P = affine_transform(Lab_dataset[file_int], M, translation, output_shape=(64, 64, 64), order=1)
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "blured"
    # -------------------------------------Random Crop and Rezize-----------------------------------------------
    if aug_int == 6:
        print("---Crop & Resize---")
        i = 10  # start at layer 10
        K = np.empty(Block_Size)
        P = np.empty(Block_Size)
        for n in range(Img_dataset[file_int].shape[2]):         # go through each nifty slice
            crop_k = Img_dataset[file_int][16:48,16:48, n]
            crop_p = Img_dataset[file_int][16:48, 16:48, n]
            crop_resize_k = resize(crop_k, (64,64))
            crop_resize_p = resize(crop_p, (64,64))
            K[:, :, n] = crop_resize_k                            # append skewed slice to numpy array (image)
            P[:, :, n] = crop_resize_p
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        aug_str = "crop & resize"
    # -------------------------------------NO AUGMENTATION-----------------------------------------------
    if aug_int == 7:
        print("---no augmentation---")
        i = 10  # start at layer 10
        K = Img_dataset[file_int]
        P = Lab_dataset[file_int]
        print(K.shape)
        plt.imshow(K[:, i])
        plt.show()
        i = i + 1
        return ()  # dont save unaugmented image, therefore leave here

    new_image = nib.Nifti1Image(K, affine=np.eye(4))
    nib.save(new_image, Img_str_lst[file_int] + '_' + aug_str + '.nii')
    new_label = nib.Nifti1Image(P, affine=np.eye(4))
    nib.save(new_label, Lab_str_lst[file_int] + '_' + aug_str + '.nii')


for files in range(len(Img_dataset)):
    Aug_rand = rm.randrange(len(Aug_Whtlst))
    print(Aug_rand)
    print(Aug_Whtlst[Aug_rand])
    print(files)
    augmentation(6, files)




