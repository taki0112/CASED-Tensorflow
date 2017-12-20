import SimpleITK as sitk
import numpy as np
import csv
import os, glob
from PIL import Image
import matplotlib.pyplot as plt

import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import block_reduce

OUTPUT_SPACING = [1.25, 1.25, 1.25]
patch_size = 68
label_size = 8
stride = 8

move = (patch_size // 2) // stride  # 4

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def read_csv(filename):
    lines = []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)

    lines = lines[1:] # remove csv headers
    annotations_dict = {}
    for i in lines:
        series_uid, x, y, z, diameter = i
        value = {'position':[float(x),float(y),float(z)],
                 'diameter':float(diameter)}
        if series_uid in annotations_dict.keys():
            annotations_dict[series_uid].append(value)
        else:
            annotations_dict[series_uid] = [value]

    return annotations_dict

def compute_coord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def normalize_planes(npzarray):
    maxHU = 400. # 600
    minHU = -1000. # -1200

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN

    return image

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def get_patch(image, coords, offset, nodule_list, patch_flag=True):
    xyz = image[int(coords[0] - offset): int(coords[0] + offset), int(coords[1] - offset): int(coords[1] + offset),
          int(coords[2] - offset): int(coords[2] + offset)]

    if patch_flag:
        output = np.expand_dims(xyz, axis=-1)
        print(coords, np.shape(output))
    else:
        # resize xyz
        """
        xyz = scipy.ndimage.zoom(input=xyz, zoom=1/8, order=1) # nearest
        xyz = np.where(xyz > 0, 1.0, 0.0)
        """
        xyz = block_reduce(xyz, (9, 9, 9), np.max)
        output = np.expand_dims(xyz, axis=-1)

        output = indices_to_one_hot(output.astype(np.int32), 2)
        output = np.reshape(output, (label_size, label_size, label_size, 2))
        output = output.astype(np.float32)

        # print('------------------')
        # print(output)

        # print(output)
        # print(np.shape(output))

    nodule_list.append(output)

def patch_stride(image, coords, offset, nodule_list, patch_flag=True):
    # get stride * 7
    # In this case, get 8*7 = 56
    original_coords = coords
    move_stride = stride

    # center
    get_patch(image, coords, offset, nodule_list, patch_flag)

    # x
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[0] = coords[0] + move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

        coords[0] = coords[0] - move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

    # y
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[1] = coords[1] + move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

        coords[1] = coords[1] - move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

    # z
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[2] = coords[2] + move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

        coords[2] = coords[2] - move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

    # xy
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[0] = coords[0] + move_stride
        coords[1] = coords[1] + move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

        coords[0] = coords[0] - move_stride
        coords[1] = coords[1] - move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

    # xz
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[0] = coords[0] + move_stride
        coords[2] = coords[2] + move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

        coords[0] = coords[0] - move_stride
        coords[2] = coords[2] - move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

    # yz
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[1] = coords[1] + move_stride
        coords[2] = coords[2] + move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

        coords[1] = coords[1] - move_stride
        coords[2] = coords[2] - move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

    # xyz
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords += move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

        coords -= move_stride
        get_patch(image, coords, offset, nodule_list, patch_flag)
        coords = original_coords

def resample(image, org_spacing, new_spacing):

    resize_factor = org_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = org_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def create_label(arr_shape, nodules, origin, new_spacing, coord=False):
    """
        nodules = list of dict {'position', 'diameter'}
    """

    def _create_mask(arr_shape, position, diameter):

        z_dim, y_dim, x_dim = arr_shape
        z_pos, y_pos, x_pos = position

        z,y,x = np.ogrid[-z_pos:z_dim-z_pos, -y_pos:y_dim-y_pos, -x_pos:x_dim-x_pos]
        mask = z**2 + y**2 + x**2 <= int(diameter//2)**2

        return mask

    if coord:
        label = []
    else:
        label = np.zeros(arr_shape, dtype='bool')

    for nodule in nodules:
        worldCoord = nodule['position']
        worldCoord = np.asarray([worldCoord[2],worldCoord[1],worldCoord[0]])

        # new_spacing came from resample
        voxelCoord = compute_coord(worldCoord, origin, new_spacing)
        voxelCoord = [int(i) for i in voxelCoord]

        diameter = nodule['diameter']
        diameter = diameter / new_spacing[1]

        if coord:
            label.append(voxelCoord + [diameter])
        else:
            mask = _create_mask(arr_shape, voxelCoord, diameter)
            label = np.logical_or(label, mask)

    return label


def plot(image, label, z_idx):

    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    ax.imshow(image[z_idx,:,:],cmap='gray')

    ax = fig.add_subplot(1,2,2)
    ax.imshow(label[z_idx,:,:],cmap='gray')

    fig.show()
