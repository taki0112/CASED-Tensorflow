import SimpleITK as sitk
import numpy as np
import csv
import os, glob
from PIL import Image
import matplotlib.pyplot as plt
import pickle

import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from time import time

OUTPUT_SPACING = [1.25, 1.25, 1.25]
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
    #maxHU = 600.
    #minHU = -1200.
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def zero_center(image):
    PIXEL_MEAN = 0.25
    image = image - PIXEL_MEAN

    return image

def resample(image, org_spacing, new_spacing=OUTPUT_SPACING):

    resize_factor = org_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = org_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def create_label(arr_shape, nodules, new_spacing, coord=False):
    """
        nodules = list of dict {'position', 'diameter'}
    """
    # https://code.i-harness.com/ko/q/83f170
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

if __name__=="__main__":

    from matplotlib.patches import Circle

    coord = True
    exclude_flag = True

    dst_spacing = OUTPUT_SPACING
    #src_root = '/lunit/data/LUNA16/rawdata'
    dst_root = '/data2/jhkim/npydata'

    src_root = '/data/jhkim/LUNA16/original'


    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    if exclude_flag :
        annotation_csv = os.path.join(src_root,'CSVFILES/annotations_excluded.csv')
    else :
        annotation_csv = os.path.join(src_root,'CSVFILES/annotations.csv')

    annotations = read_csv(annotation_csv)

    src_mhd = []
    coord_dict = {}
    all_file = len(glob.glob(os.path.join(src_root,'subset[0-9]','*.mhd')))
    cnt = 1

    for i in glob.glob(os.path.join(src_root,'subset[0-9]','*.mhd')):
        st = time()
        filename = os.path.split(i)[-1]
        series_uid = os.path.splitext(filename)[0]

        subset_num = i.split('/')[-2]
        dst_subset_path = os.path.join(dst_root,subset_num)

        if not os.path.exists(dst_subset_path):
            os.makedirs(dst_subset_path)

        np_img, origin, spacing = load_itk_image(i)

        resampled_img, new_spacing = resample(np_img, spacing, dst_spacing)
        resampled_img_shape = resampled_img.shape

        norm_img = normalize_planes(resampled_img)
        norm_img = zero_center(norm_img)

        try:
            nodules = annotations[series_uid]
            label = create_label(resampled_img_shape, nodules, new_spacing, coord=coord)
        except:
            if coord:
                label = []
            else:
                label = np.zeros(resampled_img_shape, dtype='bool')

        # np.save(os.path.join(dst_subset_path,series_uid+'.npy'), norm_img)
        # np.save(os.path.join(dst_subset_path,series_uid+'.label.npy'), label)

        coord_dict[series_uid] = label
        with open('exclude.pkl', 'wb') as f:
           pickle.dump(coord_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('{} / {} / {}'.format(cnt, all_file, time() - st))
        cnt += 1



