import glob
import os
from random import randint
import h5py
import numpy as np
import pandas as pd
from utils import load_itk, world_2_voxel
import scipy.ndimage
from skimage.measure import block_reduce

OUTPUT_SPACING = [1.25, 1.25, 1.25]
patch_size = 68
label_size = 8
stride = 8

move = (patch_size // 2) // stride  # 4
annotations = pd.read_csv('/data/jhkim/LUNA16/CSVFILES/annotations.csv')  # save dict format may be..


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def normalize(image):
    maxHU = 400.
    minHU = -1000.

    image = (image - minHU) / (maxHU - minHU)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN

    return image


def get_patch(image, coords, offset, nodule_list, patch_flag=True):
    xyz = image[int(coords[0] - offset): int(coords[0] + offset), int(coords[1] - offset): int(coords[1] + offset),
          int(coords[2] - offset): int(coords[2] + offset)]

    if patch_flag:
        output = np.expand_dims(xyz, axis=-1)
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


def process_image(image_path, annotations, nodule, non_nodule, nodule_label, non_nodule_label):
    image, origin, spacing = load_itk(image_path)  # 512 512 119
    image_name = os.path.split(image_path)[1].replace('.mhd', '')

    subset_name = image_path.split('/')[-2]
    SH_path = '/data2/jhkim/npydata/' + subset_name + '/' + image_name + '.npy'
    label_name = '/data2/jhkim/npydata/' + subset_name + '/' + image_name + '.label.npy'

    # calculate resize factor
    resize_factor = spacing / OUTPUT_SPACING
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / image.shape
    new_spacing = spacing / real_resize

    image = np.transpose(np.load(SH_path))
    label = np.transpose(np.load(label_name))

    # image = normalize(image)
    # image = zero_center(image)

    # padding
    offset = patch_size // 2

    non_pad = image
    non_label_pad = label

    non_pad = np.pad(non_pad, offset, 'constant', constant_values=np.min(non_pad))
    non_label_pad = np.pad(non_label_pad, offset, 'constant', constant_values=np.min(non_label_pad))

    image = np.pad(image, offset + (stride * move), 'constant', constant_values=np.min(image))
    label = np.pad(label, offset + (stride * move), 'constant', constant_values=np.min(label))

    indices = annotations[annotations['seriesuid'] == image_name].index

    nodule_list = []
    nodule_label_list = []
    for i in indices:
        row = annotations.iloc[i]
        world_coords = np.array([row.coordX, row.coordY, row.coordZ])

        coords = np.floor(world_2_voxel(world_coords, origin, new_spacing)) + offset + (stride * move)  # center
        patch_stride(image, coords, offset, nodule_list)  # x,y,z, xy,xz,yz, xyz ... get stride patch
        patch_stride(label, coords, offset, nodule_label_list, patch_flag=False)

    nodule_num = len(nodule_list)

    non_nodule_list = []
    non_nodule_label_list = []
    x_coords = non_pad.shape[0] - offset - 1
    y_coords = non_pad.shape[1] - offset - 1
    z_coords = non_pad.shape[2] - offset - 1

    while len(non_nodule_list) < 3 * nodule_num:
        rand_x = randint(offset, x_coords)
        rand_y = randint(offset, y_coords)
        rand_z = randint(offset, z_coords)

        coords = np.array([rand_x, rand_y, rand_z])

        get_patch(non_pad, coords, offset, non_nodule_list)
        get_patch(non_label_pad, coords, offset, non_nodule_label_list, patch_flag=False)

    nodule.extend(nodule_list)
    non_nodule.extend(non_nodule_list)
    nodule_label.extend(nodule_label_list)
    non_nodule_label.extend(non_nodule_label_list)

    print('nodule : ', np.shape(nodule))
    print('nodule_label : ', np.shape(nodule_label))
    print('non-nodule : ', np.shape(non_nodule))
    print('non-nodule_label : ', np.shape(non_nodule_label))


for i in range(10):
    image_paths = glob.glob("/data/jhkim/LUNA16/original/subset" + str(i) + '/*.mhd')
    nodule = []
    non_nodule = []
    nodule_label = []
    non_nodule_label = []

    idx = 1
    flag = 1
    save_path = '/data2/jhkim/LUNA16/patch/SH/'
    
    for image_path in image_paths:
        print('subset' + str(i) + ' / ' + str(idx) + ' / ' + str(len(image_paths)))
        process_image(image_path, annotations, nodule, non_nodule, nodule_label, non_nodule_label)
        idx += 1

    np.random.seed(0)
    np.random.shuffle(nodule)
    np.random.seed(0)
    np.random.shuffle(nodule_label)

    np.random.seed(0)
    np.random.shuffle(non_nodule)
    np.random.seed(0)
    np.random.shuffle(non_nodule_label)

    train_nodule_len = int(len(nodule) * 0.7)
    train_non_nodule_len = int(len(non_nodule) * 0.7)

    # print(np.shape(nodule))
    # print(np.shape(non_nodule))

    # print(np.shape(nodule[:train_nodule_len]))

    with h5py.File(save_path + 'subset' + str(i) + '.h5', 'w') as hf:
        hf.create_dataset('nodule', data=nodule[:], compression='lzf')
        hf.create_dataset('label_nodule', data=nodule_label[:], compression='lzf')

        hf.create_dataset('non_nodule', data=non_nodule[:], compression='lzf')
        hf.create_dataset('label_non_nodule', data=non_nodule_label[:], compression='lzf')

    # with h5py.File(save_path + 't_subset' + str(i) + '.h5', 'w') as hf:
    #     hf.create_dataset('nodule', data=nodule[train_nodule_len:], compression='lzf')
    #     hf.create_dataset('label_nodule', data=nodule_label[train_nodule_len:], compression='lzf')
    #
    #     hf.create_dataset('non_nodule', data=non_nodule[train_non_nodule_len:], compression='lzf')
    #     hf.create_dataset('label_non_nodule', data=non_nodule_label[train_non_nodule_len:], compression='lzf')
