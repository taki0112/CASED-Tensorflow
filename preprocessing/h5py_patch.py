import glob, os, h5py, scipy
from random import randint
import numpy as np
import pandas as pd
from utils import load_itk, world_2_voxel


OUTPUT_SPACING = [1.25, 1.25, 1.25]
patch_size = 68
stride = 8

move = (patch_size // 2) // stride # 4
annotations = pd.read_csv('/data/jhkim/LUNA16/CSVFILES/annotations.csv')  # save dict format may be..
def normalize(image):
    maxHU = 400.
    minHU = -1000.

    image = (image - minHU) / (maxHU - minHU)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def zero_center(image) :
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN

    return image

def get_patch(image, coords, offset, nodule_list):
    xyz = image[int(coords[0] - offset): int(coords[0] + offset), int(coords[1] - offset): int(coords[1] + offset), int(coords[2] - offset): int(coords[2] + offset)]
    xyz = normalize(xyz)
    xyz = zero_center(xyz)
    output = np.zeros([patch_size, patch_size, patch_size, 1])
    output[ :, :, :, 0] = xyz

    nodule_list.append(output)

def patch_stride(image, coords, offset, nodule_list):
    # get stride * 7
    # In this case, get 8*7 = 56
    original_coords = coords
    move_stride = stride

    # center
    get_patch(image, coords, offset, nodule_list)

    # x
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[0] = coords[0] + move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

        coords[0] = coords[0] - move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

    # y
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[1] = coords[1] + move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

        coords[1] = coords[1] - move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

    # z
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[2] = coords[2] + move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

        coords[2] = coords[2] - move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

    # xy
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[0] = coords[0] + move_stride
        coords[1] = coords[1] + move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

        coords[0] = coords[0] - move_stride
        coords[1] = coords[1] - move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

    # xz
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[0] = coords[0] + move_stride
        coords[2] = coords[2] + move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

        coords[0] = coords[0] - move_stride
        coords[2] = coords[2] - move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

    # yz
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords[1] = coords[1] + move_stride
        coords[2] = coords[2] + move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

        coords[1] = coords[1] - move_stride
        coords[2] = coords[2] - move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

    # xyz
    for idx in range(1, move + 1):
        move_stride = stride * idx
        coords += move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

        coords -= move_stride
        get_patch(image, coords, offset, nodule_list)
        coords = original_coords

def process_image(image_path, annotations, nodule, non_nodule) :
    image, origin, spacing = load_itk(image_path) # 512 512 119


    #calculate resize factor
    resize_factor = spacing / OUTPUT_SPACING
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize = new_shape / image.shape
    new_spacing = spacing / real_resize

    #resize image
    image = scipy.ndimage.interpolation.zoom(image, real_resize) # 304 304 238

    # padding
    offset = patch_size // 2

    non_pad = image
    non_pad = np.pad(non_pad, offset, 'constant', constant_values=0)

    image = np.pad(image, offset + (stride*move), 'constant', constant_values=0)

    image_name = os.path.split(image_path)[1].replace('.mhd', '')

    indices = annotations[annotations['seriesuid'] == image_name].index

    nodule_list = []
    for i in indices :
        row = annotations.iloc[i]
        world_coords = np.array([row.coordX, row.coordY, row.coordZ])

        coords = np.floor(world_2_voxel(world_coords, origin, new_spacing)) + offset # center
        patch_stride(image, coords, offset, nodule_list) # x,y,z, xy,xz,yz, xyz ... get stride patch

    nodule_num = len(nodule_list)

    non_nodule_list = []
    x_coords = non_pad.shape[0] - offset - 1
    y_coords = non_pad.shape[1] - offset - 1
    z_coords = non_pad.shape[2] - offset - 1

    while len(non_nodule_list) < 3 * nodule_num :
        rand_x = randint(offset, x_coords)
        rand_y = randint(offset, y_coords)
        rand_z = randint(offset, z_coords)

        coords = np.array([rand_x, rand_y, rand_z])

        get_patch(non_pad, coords, offset, non_nodule_list)

    nodule.extend(nodule_list)
    non_nodule.extend(non_nodule_list)

    print('nodule : ', np.shape(nodule))
    print('non-nodule : ', np.shape(non_nodule))


for i in range(10) :
    image_paths = glob.glob("/data/jhkim/LUNA16/original/subset" + str(i) + '/*.mhd')
    nodule = []
    non_nodule = []
    idx = 1
    flag = 1
    save_path = '/data2/jhkim/LUNA16/patch/split/'
    for image_path in image_paths :
        print('subset' + str(i) + ' / ' + str(idx) + ' / ' + str(len(image_paths)))
        process_image(image_path, annotations, nodule, non_nodule)
        idx += 1
    np.random.shuffle(nodule)
    np.random.shuffle(non_nodule)

    train_nodule_len = int(len(nodule) * 0.7)
    train_non_nodule_len = int(len(non_nodule) * 0.7)

    with h5py.File(save_path + 'subset' + str(i) + '.h5', 'w') as hf:
        hf.create_dataset('nodule', data=nodule[:train_nodule_len], compression='lzf')
        hf.create_dataset('non_nodule', data=non_nodule[:train_non_nodule_len], compression='lzf')

    with h5py.File(save_path + 't_subset' + str(i) + '.h5', 'w') as hf:
        hf.create_dataset('nodule', data=nodule[train_nodule_len:], compression='lzf')
        hf.create_dataset('non_nodule', data=non_nodule[train_non_nodule_len:], compression='lzf')
