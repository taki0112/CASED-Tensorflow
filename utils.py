import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import os, h5py, glob
import tensorflow.contrib.slim as slim
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.ticker import  FixedFormatter
import matplotlib


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
        # print(np.shape(nodule))
    return nodule


def non_nodule_hf(idx):
    with h5py.File(file, 'r') as hf:
        non_nodule = hf['non_nodule'][idx:idx + get_data_num]
    return non_nodule

def label_nodule_hf(idx):
    with h5py.File(file, 'r') as hf:
        nodule = hf['label_nodule'][idx:idx + get_data_num]
        # print(np.shape(nodule))
    return nodule


def label_non_nodule_hf(idx):
    with h5py.File(file, 'r') as hf:
        non_nodule = hf['label_non_nodule'][idx:idx + get_data_num]
    return non_nodule


def prepare_data(sub_n):
    global file
    dir = '/data2/jhkim/LUNA16/patch/SH/subset'
    file = ''.join([dir, str(sub_n), '.h5'])

    with h5py.File(file, 'r') as fin:
        nodule_range = range(0, len(fin['nodule']), get_data_num)
        non_nodule_range = range(0, len(fin['non_nodule']), get_data_num)
        label_nodule_range = range(0, len(fin['label_nodule']), get_data_num)
        label_non_nodule_range = range(0, len(fin['label_non_nodule']), get_data_num)


    pool = Pool(processes=process_num)
    pool_nodule = pool.map(nodule_hf, nodule_range)
    pool_non_nodule = pool.map(non_nodule_hf, non_nodule_range)
    pool_label_nodule = pool.map(label_nodule_hf, label_nodule_range)
    pool_label_non_nodule = pool.map(label_non_nodule_hf, label_non_nodule_range)

    pool.close()

    # print(np.shape(nodule[0]))
    nodule = []
    non_nodule = []

    nodule_y = []
    non_nodule_y = []

    for p in pool_nodule:
        nodule.extend(p)
    for p in pool_non_nodule:
        non_nodule.extend(p)

    for p in pool_label_nodule:
        nodule_y.extend(p)
    for p in pool_label_non_nodule:
        non_nodule_y.extend(p)

    nodule = np.asarray(nodule)
    non_nodule = np.asarray(non_nodule)
    nodule_y = np.asarray(nodule_y)
    non_nodule_y = np.asarray(non_nodule_y)

    #print(np.shape(nodule_y))
    #print(np.shape(non_nodule_y))

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

def validation_data(sub_n) :
    global file
    dir = '/data2/jhkim/LUNA16/patch/SH/subset'
    file = ''.join([dir, str(sub_n), '.h5'])

    with h5py.File(file, 'r') as fin:
        nodule_range = range(0, len(fin['nodule']), get_data_num)
        non_nodule_range = range(0, len(fin['non_nodule']), get_data_num)
        label_nodule_range = range(0, len(fin['label_nodule']), get_data_num)
        label_non_nodule_range = range(0, len(fin['label_non_nodule']), get_data_num)


    pool = Pool(processes=process_num)
    pool_nodule = pool.map(nodule_hf, nodule_range)
    pool_non_nodule = pool.map(non_nodule_hf, non_nodule_range)
    pool_label_nodule = pool.map(label_nodule_hf, label_nodule_range)
    pool_label_non_nodule = pool.map(label_non_nodule_hf, label_non_nodule_range)

    pool.close()

    # print(np.shape(nodule[0]))
    nodule = []
    non_nodule = []

    nodule_y = []
    non_nodule_y = []

    for p in pool_nodule:
        nodule.extend(p)
    for p in pool_non_nodule:
        non_nodule.extend(p)

    for p in pool_label_nodule:
        nodule_y.extend(p)
    for p in pool_label_non_nodule:
        non_nodule_y.extend(p)

    nodule = np.asarray(nodule)
    non_nodule = np.asarray(non_nodule)
    nodule_y = np.asarray(nodule_y)
    non_nodule_y = np.asarray(non_nodule_y)

    all_y = np.concatenate([nodule_y, non_nodule_y], axis=0)
    all_patch = np.concatenate([nodule, non_nodule], axis=0)

    # seed = 777
    # np.random.seed(seed)
    # np.random.shuffle(all_patch)
    # np.random.seed(seed)
    # np.random.shuffle(all_y)


    return all_patch, all_y

def test_data(sub_n):
    global file
    dir = '/data2/jhkim/LUNA16/patch/SH/t_subset'
    file = ''.join([dir, str(sub_n), '.h5'])

    with h5py.File(file, 'r') as fin:
        nodule_range = range(0, len(fin['nodule']), get_data_num)
        non_nodule_range = range(0, len(fin['non_nodule']), get_data_num)
        label_nodule_range = range(0, len(fin['label_nodule']), get_data_num)
        label_non_nodule_range = range(0, len(fin['label_non_nodule']), get_data_num)


    pool = Pool(processes=process_num)
    pool_nodule = pool.map(nodule_hf, nodule_range)
    pool_non_nodule = pool.map(non_nodule_hf, non_nodule_range)
    pool_label_nodule = pool.map(label_nodule_hf, label_nodule_range)
    pool_label_non_nodule = pool.map(label_non_nodule_hf, label_non_nodule_range)

    pool.close()

    # print(np.shape(nodule[0]))
    nodule = []
    non_nodule = []

    nodule_y = []
    non_nodule_y = []

    for p in pool_nodule:
        nodule.extend(p)
    for p in pool_non_nodule:
        non_nodule.extend(p)

    for p in pool_label_nodule:
        nodule_y.extend(p)
    for p in pool_label_non_nodule:
        non_nodule_y.extend(p)

    nodule = np.asarray(nodule)
    non_nodule = np.asarray(non_nodule)
    nodule_y = np.asarray(nodule_y)
    non_nodule_y = np.asarray(non_nodule_y)

    all_y = np.concatenate([nodule_y, non_nodule_y], axis=0)
    all_patch = np.concatenate([nodule, non_nodule], axis=0)

    # seed = 777
    # np.random.seed(seed)
    # np.random.shuffle(all_patch)
    # np.random.seed(seed)
    # np.random.shuffle(all_y)


    return all_patch, all_y


def sensitivity(logits, labels):
    predictions = tf.argmax(logits, axis=-1)
    actuals = tf.argmax(labels, axis=-1)


    nodule_actuals = tf.ones_like(actuals)
    non_nodule_actuals = tf.zeros_like(actuals)
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

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, nodule_actuals),
                tf.equal(predictions, non_nodule_predictions)
            ),
            tf.float32
        )
    )

    false_positive_rate = fp_op / (fp_op + tn_op)

    recall = tp_op / (tp_op + fn_op)

    return recall, false_positive_rate

def Snapshot(t, T, M, alpha_zero) :
    """

    t = # of current iteration
    T = # of total iteration
    M = # of snapshot
    alpha_zero = init learning rate

    """

    x = (np.pi * (t % (T // M))) / (T // M)
    x = np.cos(x) + 1

    lr = (alpha_zero / 2) * x

    return lr

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def create_exclude_mask(arr_shape, position, diameter):
    x_dim, y_dim, z_dim = arr_shape
    x_pos, y_pos, z_pos = position

    x, y, z = np.ogrid[-x_pos:x_dim - x_pos, -y_pos:y_dim - y_pos, -z_pos:z_dim - z_pos]
    mask = x ** 2  + y ** 2 + z ** 2 > int(diameter // 2) ** 2

    return mask

def fp_per_scan(logit, label) :
    logit = np.reshape(logit, -1)
    label = np.reshape(label, -1)

    logit = logit[logit >= 0]
    label = label[label >= 0]

    logit = np.where(logit >= 1.0, logit-1, logit)
    label = np.where(label >= 1.0, label-1, label)

    fpr, tpr, th = roc_curve(label, logit, pos_label=1.0)
    negative_samples = np.count_nonzero(label == 0.0)
    fps = fpr * negative_samples


    """
    mean_sens = np.mean(sens_list)
    matplotlib.use('Agg')

    ax = plt.gca()
    plt.plot(fps_itp, sens_itp)
    # https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.grid.html
    plt.xlim(MIN_FROC, MAX_FROC)
    plt.ylim(0, 1.1)
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    # plt.legend(loc='lower right')
    # plt.legend(loc=9)
    plt.title('Average sensitivity = %.4f' % (mean_sens))

    plt.xscale('log', basex=2)
    ax.xaxis.set_major_formatter(FixedFormatter(fp_list))

    ax.xaxis.set_ticks(fp_list)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=True, linestyle='dotted', which='both')
    plt.tight_layout()

    # plt.show()
    plt.savefig('result.png', bbox_inches=0, dpi=300)
    """

    return np.asarray(fps), np.asarray(tpr)