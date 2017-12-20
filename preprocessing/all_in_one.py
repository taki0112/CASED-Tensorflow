from preprocessing.preprocess_utils import *
from random import randint
import h5py
OUTPUT_SPACING = [1.25, 1.25, 1.25]
patch_size = 68
label_size = 8
stride = 8
move = (patch_size // 2) // stride  # 4


if __name__=="__main__":

    from matplotlib.patches import Circle

    coord = False

    dst_spacing = OUTPUT_SPACING
    src_root = '/data/jhkim/LUNA16/original'
    #dst_root = '/lunit/data/LUNA16/npydata'

    save_path = '/data2/jhkim/LUNA16/patch/SH/'

    # if not os.path.exists(dst_root):
    #     os.makedirs(dst_root)

    annotation_csv = os.path.join(src_root,'CSVFILES/annotations.csv')
    annotations = read_csv(annotation_csv)

    src_mhd = []
    coord_dict = {}
    for sub_n in range(10) :
        idx = 1
        sub_n_str = 'subset' + str(sub_n)
        image_paths = glob.glob(os.path.join(src_root,sub_n_str,'*.mhd'))

        nodule = []
        non_nodule = []
        nodule_label = []
        non_nodule_label = []

        for i in image_paths:
            filename = os.path.split(i)[-1]
            series_uid = os.path.splitext(filename)[0]

            subset_num = i.split('/')[-2]
            # dst_subset_path = os.path.join(dst_root,subset_num)
            #
            # if not os.path.exists(dst_subset_path):
            #     os.makedirs(dst_subset_path)

            np_img, origin, spacing = load_itk_image(i)

            resampled_img, new_spacing = resample(np_img, spacing, dst_spacing)
            resampled_img_shape = resampled_img.shape

            norm_img = normalize_planes(resampled_img)
            norm_img = zero_center(norm_img)
            norm_img_shape = norm_img.shape
            nodule_coords = []
            try:
                nodule_coords = annotations[series_uid]
                label = create_label(resampled_img_shape, nodule_coords, origin, new_spacing, coord=coord)
            except:
                if coord:
                    label = []
                else:
                    label = np.zeros(resampled_img_shape, dtype='bool')

            image = np.transpose(norm_img)
            label = np.transpose(label)
            #origin = np.transpose(origin)
            #new_spacing = np.transpose(new_spacing)

            #np.save(os.path.join(dst_subset_path,series_uid+'.npy'), norm_img)
            #np.save(os.path.join(dst_subset_path,series_uid+'.label.npy'), label)
            #coord_dict[os.path.join(dst_subset_path,series_uid+'.npy')] = label

            # padding
            offset = patch_size // 2

            non_pad = image
            non_label_pad = label

            non_pad = np.pad(non_pad, offset, 'constant', constant_values=0)
            non_label_pad = np.pad(non_label_pad, offset, 'constant', constant_values=0)

            image = np.pad(image, offset + (stride * move), 'constant', constant_values=0)
            label = np.pad(label, offset + (stride * move), 'constant', constant_values=0)

            nodule_list = []
            nodule_label_list = []

            for nodule_coord in nodule_coords :
                worldCoord = nodule_coord['position']
                worldCoord = np.asarray([worldCoord[2], worldCoord[1], worldCoord[0]])

                # new_spacing came from resample
                voxelCoord = compute_coord(worldCoord, origin, new_spacing)
                voxelCoord = np.asarray([int(i) + offset + (stride * move) for i in voxelCoord])
                voxelCoord = np.transpose(voxelCoord)

                patch_stride(image, voxelCoord, offset, nodule_list)  # x,y,z, xy,xz,yz, xyz ... get stride patch
                patch_stride(label, voxelCoord, offset, nodule_label_list, patch_flag=False)

            #print(series_uid)

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

            print(sub_n_str + ' / ' + str(idx) + ' / ' + str(len(image_paths)))
            print('nodule : ', np.shape(nodule))
            print('nodule_label : ', np.shape(nodule_label))
            print('non-nodule : ', np.shape(non_nodule))
            print('non-nodule_label : ', np.shape(non_nodule_label))
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

        with h5py.File(save_path + 'subset' + str(sub_n) + '.h5', 'w') as hf:
            hf.create_dataset('nodule', data=nodule[:], compression='lzf')
            hf.create_dataset('label_nodule', data=nodule_label[:], compression='lzf')

            hf.create_dataset('non_nodule', data=non_nodule[:], compression='lzf')
            hf.create_dataset('label_non_nodule', data=non_nodule_label[:], compression='lzf')



        # with h5py.File(save_path + 'subset' + str(sub_n) + '.h5', 'w') as hf:
        #     hf.create_dataset('nodule', data=nodule[:train_nodule_len], compression='lzf')
        #     hf.create_dataset('label_nodule', data=nodule_label[:train_nodule_len], compression='lzf')
        #
        #     hf.create_dataset('non_nodule', data=non_nodule[:train_non_nodule_len], compression='lzf')
        #     hf.create_dataset('label_non_nodule', data=non_nodule_label[:train_non_nodule_len], compression='lzf')
        #
        # with h5py.File(save_path + 't_subset' + str(sub_n) + '.h5', 'w') as hf:
        #     hf.create_dataset('nodule', data=nodule[train_nodule_len:], compression='lzf')
        #     hf.create_dataset('label_nodule', data=nodule_label[train_nodule_len:], compression='lzf')
        #
        #     hf.create_dataset('non_nodule', data=non_nodule[train_non_nodule_len:], compression='lzf')
        #     hf.create_dataset('label_non_nodule', data=non_nodule_label[train_non_nodule_len:], compression='lzf')


