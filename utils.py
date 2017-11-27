import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import os, h5py
import tensorflow.contrib.slim as slim
from multiprocessing import Pool
seed = 0
file = ""

process_num = 32
get_data_num = 64

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image = np.transpose(sitk.GetArrayFromImage(itkimage))
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    return image, origin, spacing

def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def voxel_2_world(voxel_coord, origin, spacing):
    stretched_voxel_coord = voxel_coord * spacing
    world_coord = stretched_voxel_coord + origin
    return world_coord

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def nodule_hf(idx):
    with h5py.File(file, 'r') as hf:
        nodule = hf['nodule'][idx:idx + get_data_num]
    return nodule


def non_nodule_hf(idx):
    with h5py.File(file, 'r') as hf:
        non_nodule = hf['non_nodule'][idx:idx + get_data_num]
    return non_nodule

def prepare_date(sub_n):
    global file
    dir = '/data2/jhkim/LUNA16/patch/split/subset'
    file = ''.join([dir,str(sub_n),'.h5'])

    with h5py.File(file, 'r') as fin :
        nodule_range = range(0, len(fin['nodule']), get_data_num)
        non_nodule_range = range(0, len(fin['non_nodule']), get_data_num)
    pool = Pool(processes=process_num)
    pool_nodule = pool.map(nodule_hf, nodule_range)
    pool_non_nodule = pool.map(non_nodule_hf, non_nodule_range)

    pool.close()

    nodule = []
    non_nodule = []
    for p in pool_nodule :
        nodule.extend(p)
    for p in pool_non_nodule :
        non_nodule.extend(p)

    nodule = np.asarray(nodule)
    non_nodule = np.asarray(non_nodule)

    non_nodule_y = np.zeros(shape=(len(non_nodule), 8, 8, 8, 2), dtype=np.float)
    non_nodule_y[:, :, :, :, 0] = 1.0

    nodule_y = np.zeros(shape=(len(nodule), 8, 8, 8, 2), dtype=np.float)
    nodule_y[:, :, :, :, 1] = 1.0


    all_y = np.concatenate([nodule_y, non_nodule_y], axis=0)
    all_patch = np.concatenate([nodule, non_nodule], axis=0)


    # seed = 777
    # np.random.seed(seed)
    # np.random.shuffle(nodule)
    # np.random.seed(seed)
    # np.random.shuffle(nodule_y)
    #
    # np.random.seed(seed)
    # np.random.shuffle(all_patch)
    # np.random.seed(seed)
    # np.random.shuffle(all_y)


    return nodule, all_patch, nodule_y, all_y

def test_data(sub_n):
    global file
    dir = '/data2/jhkim/LUNA16/patch/split/t_subset'
    file = ''.join([dir, str(sub_n), '.h5'])

    with h5py.File(file, 'r') as fin :
        nodule_range = range(0, len(fin['nodule']), get_data_num)
        non_nodule_range = range(0, len(fin['non_nodule']), get_data_num)
    pool = Pool(processes=process_num)
    pool_nodule = pool.map(nodule_hf, nodule_range)
    pool_non_nodule = pool.map(non_nodule_hf, non_nodule_range)

    pool.close()

    nodule = []
    non_nodule = []
    for p in pool_nodule :
        nodule.extend(p)
    for p in pool_non_nodule :
        non_nodule.extend(p)
    
    nodule = np.asarray(nodule)
    non_nodule = np.asarray(non_nodule)

    non_nodule_y = np.zeros(shape=(len(non_nodule), 8, 8, 8, 2), dtype=np.float)
    non_nodule_y[:, :, :, :, 0] = 1.0

    nodule_y = np.zeros(shape=(len(nodule), 8, 8, 8, 2), dtype=np.float)
    nodule_y[:, :, :, :, 1] = 1.0


    all_y = np.concatenate([nodule_y, non_nodule_y], axis=0)
    all_patch = np.concatenate([nodule, non_nodule], axis=0)

    # seed = 777
    # np.random.seed(seed)
    # np.random.shuffle(all_patch)
    # np.random.seed(seed)
    # np.random.shuffle(all_y)


    return all_patch, all_y

def sensitivity(logits, labels) :
    predictions = tf.argmax(logits, axis=-1)
    actuals = tf.argmax(labels, axis=-1)



    nodule_actuals = tf.ones_like(actuals)
    # non_nodule_actuals = tf.zeros_like(actuals)
    nodule_predictions = tf.ones_like(predictions)
    non_nodule_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, nodule_actuals),
                tf.equal(predictions, nodule_predictions)
            ),
            tf.float32
        )
    )
    """
    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, non_nodule_actuals),
                tf.equal(predictions, non_nodule_predictions)
            ),
            tf.float32
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, non_nodule_actuals),
                tf.equal(predictions, nodule_predictions)
            ),
            tf.float32
        )
    )
    """
    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, nodule_actuals),
                tf.equal(predictions, non_nodule_predictions)
            ),
            tf.float32
        )
    )

    recall = tp_op / (tp_op + fn_op)

    return recall
