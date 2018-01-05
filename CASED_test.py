import time, pickle
from ops import *
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_blocks as patch_blocks
from math import ceil
from skimage.measure import block_reduce


class CASED(object):
    def __init__(self, sess, batch_size, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = 'LUNA16'
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.model_name = "CASED"  # name for checkpoint

        self.c_dim = 1
        self.y_dim = 2  # nodule ? or non_nodule ?
        self.block_size = 68

    def cased_network(self, x, reuse=False, scope='CASED_NETWORK'):
        with tf.variable_scope(scope, reuse=reuse):
            x = conv_layer(x, channels=32, kernel=3, stride=1, layer_name='conv1')
            up_conv1 = conv_layer(x, channels=32, kernel=3, stride=1, layer_name='up_conv1')

            x = max_pooling(up_conv1)

            x = conv_layer(x, channels=64, kernel=3, stride=1, layer_name='conv2')
            up_conv2 = conv_layer(x, channels=64, kernel=3, stride=1, layer_name='up_conv2')

            x = max_pooling(up_conv2)

            x = conv_layer(x, channels=128, kernel=3, stride=1, layer_name='conv3')
            up_conv3 = conv_layer(x, channels=128, kernel=3, stride=1, layer_name='up_conv3')

            x = max_pooling(up_conv3)

            x = conv_layer(x, channels=256, kernel=3, stride=1, layer_name='conv4')
            x = conv_layer(x, channels=128, kernel=3, stride=1, layer_name='conv5')

            x = deconv_layer(x, channels=128, kernel=4, stride=2, layer_name='deconv1')
            x = copy_crop(crop_layer=up_conv3, in_layer=x)

            x = conv_layer(x, channels=128, kernel=1, stride=1, layer_name='conv6')
            x = conv_layer(x, channels=64, kernel=1, stride=1, layer_name='conv7')

            x = deconv_layer(x, channels=64, kernel=4, stride=2, layer_name='deconv2')
            x = copy_crop(crop_layer=up_conv2, in_layer=x)

            x = conv_layer(x, channels=64, kernel=1, stride=1, layer_name='conv8')
            x = conv_layer(x, channels=32, kernel=1, stride=1, layer_name='conv9')

            x = deconv_layer(x, channels=32, kernel=4, stride=2, layer_name='deconv3')
            x = copy_crop(crop_layer=up_conv1, in_layer=x)

            x = conv_layer(x, channels=32, kernel=1, stride=1, layer_name='conv10')
            x = conv_layer(x, channels=32, kernel=1, stride=1, layer_name='conv11')

            logits = conv_layer(x, channels=2, kernel=1, stride=1, activation=None, layer_name='conv12')

            x = softmax(logits)

            return logits, x

    def build_model(self):

        bs = None
        scan_dims = [None, None, None, self.c_dim]
        scan_y_dims = [None, None, None, self.y_dim]

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + scan_dims, name='patch')

        # labels
        self.y = tf.placeholder(tf.float32, [bs] + scan_y_dims, name='y')  # for loss

        self.logits, self.softmax_logits = self.cased_network(self.inputs)

        """ Loss function """
        self.correct_prediction = tf.equal(tf.argmax(self.softmax_logits, -1), tf.argmax(self.y, -1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.sensitivity, self.fp_rate = sensitivity(labels=self.y, logits=self.softmax_logits)

        """ Summary """

        c_acc = tf.summary.scalar('acc', self.accuracy)
        c_recall = tf.summary.scalar('sensitivity', self.sensitivity)
        c_fp = tf.summary.scalar('false_positive', self.fp_rate)
        self.c_sum = tf.summary.merge([c_acc, c_recall, c_fp])

    def test(self):
        block_size = self.block_size
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        validation_sub_n = 8
        subset_name = 'subset' + str(validation_sub_n)
        print(subset_name)
        image_paths = glob.glob("/data/jhkim/LUNA16/original/subset" + str(validation_sub_n) + '/*.mhd')
        all_scan_num = len(image_paths)
        sens_list = []
        fps_list = []

        nan_num = 0
        cnt = 1


        MIN_FROC = 0.125
        MAX_FROC = 8

        for scan in image_paths:
            print('{} / {}'.format(cnt, len(image_paths)))
            scan_name = os.path.split(scan)[1].replace('.mhd', '')
            scan_npy = '/data2/jhkim/npydata/' + subset_name + '/' + scan_name + '.npy'
            label_npy = '/data2/jhkim/npydata/' + subset_name + '/' + scan_name + '.label.npy'

            image = np.transpose(np.load(scan_npy))
            label = np.transpose(np.load(label_npy))
            print(np.shape(image))

            if np.count_nonzero(label) == 0:
                nan_num += 1
                cnt += 1
                continue

            pad_list = []
            for i in range(3):
                if np.shape(image)[i] % block_size == 0:
                    pad_l = 0
                    pad_r = pad_l
                else:
                    q = (ceil(np.shape(image)[i] / block_size) * block_size) - np.shape(image)[i]

                    if q % 2 == 0:
                        pad_l = q // 2
                        pad_r = pad_l
                    else:
                        pad_l = q // 2
                        pad_r = pad_l + 1

                pad_list.append(pad_l)
                pad_list.append(pad_r)

            image = np.pad(image, pad_width=[[pad_list[0], pad_list[1]], [pad_list[2], pad_list[3]],
                                             [pad_list[4], pad_list[5]]],
                           mode='constant', constant_values=np.min(image))

            label = np.pad(label, pad_width=[[pad_list[0], pad_list[1]], [pad_list[2], pad_list[3]],
                                             [pad_list[4], pad_list[5]]],
                           mode='constant', constant_values=np.min(label))

            with open('jh_exclude.pkl', 'rb') as f:
                exclude_dict = pickle.load(f, encoding='bytes')

            exclude_coords = exclude_dict[scan_name]
            ex_mask = np.ones_like(image)
            for ex in exclude_coords:
                ex[0] = ex[0] + (pad_list[0] + pad_list[1]) // 2
                ex[1] = ex[1] + (pad_list[2] + pad_list[3]) // 2
                ex[2] = ex[2] + (pad_list[4] + pad_list[5]) // 2
                ex_diameter = ex[3]
                if ex_diameter < 0.0:
                    ex_diameter = 10.0
                exclude_position = (ex[0], ex[1], ex[2])
                exclude_mask = create_exclude_mask(image.shape, exclude_position, ex_diameter)
                ex_mask = exclude_mask if ex_mask is None else np.logical_and(ex_mask, exclude_mask)

            image_blocks = patch_blocks(image, block_shape=(block_size, block_size, block_size))
            label_blocks = patch_blocks(label, block_shape=(block_size, block_size, block_size))
            ex_mask_blocks = patch_blocks(ex_mask, block_shape=(block_size, block_size, block_size))

            len_x = len(image_blocks)
            len_y = len(image_blocks[0])
            len_z = len(image_blocks[0, 0])

            result_scan = None
            label_scan = None
            ex_scan = None
            for x_i in range(len_x):
                x = None
                x_label = None
                x_ex = None
                for y_i in range(len_y):
                    y = None
                    y_label = None
                    y_ex = None
                    for z_i in range(len_z):
                        scan = np.expand_dims(np.expand_dims(image_blocks[x_i, y_i, z_i], axis=-1), axis=0)  # 1 68 68 68 1
                        logit_label = block_reduce(label_blocks[x_i, y_i, z_i], (9, 9, 9), np.max)
                        logit_ex = block_reduce(ex_mask_blocks[x_i, y_i, z_i], (9, 9, 9), np.min)

                        test_feed_dict = {
                            self.inputs: scan
                        }

                        logits = self.sess.run(
                            self.softmax_logits, feed_dict=test_feed_dict
                        )  # [1, 68, 68, 68, 2]
                        logits_ = np.squeeze(logits, axis=0)  # [68,68,68]
                        logits = np.zeros(shape=(logits_.shape[0], logits_.shape[1], logits_.shape[2], 1))
                        for x_i_, x_v in enumerate(logits_):
                            for y_i_, y_v in enumerate(x_v):
                                for z_i_, z_v in enumerate(y_v):
                                    logits[x_i_, y_i_, z_i_] = z_v[1]
                        logits = np.squeeze(logits, axis=-1)  # 68 68 68

                        """
                        [1, 72, 72, 72, 2] -> [1, 72, 72, 72] -> [72,72,72]
                        """

                        y = logits if y is None else np.concatenate((y, logits), axis=2)  # z concat
                        y_label = logit_label if y_label is None else np.concatenate((y_label, logit_label), axis=2)
                        y_ex = logit_ex if y_ex is None else np.concatenate((y_ex, logit_ex), axis=2)

                    x = y if x is None else np.concatenate((x, y), axis=1)  # y concat
                    x_label = y_label if x_label is None else np.concatenate((x_label, y_label), axis=1)
                    x_ex = y_ex if x_ex is None else np.concatenate((x_ex, y_ex), axis=1)

                result_scan = x if result_scan is None else np.concatenate((result_scan, x), axis=0)  # x concat
                label_scan = x_label if label_scan is None else np.concatenate((label_scan, x_label), axis=0)
                ex_scan = x_ex if ex_scan is None else np.concatenate((ex_scan, x_ex), axis=0)
            label = label_scan
            ex_mask = ex_scan
            # print(result) # 3d original size

            with open('jh.pkl', 'rb') as f:
                coords_dict = pickle.load(f, encoding='bytes')

            ex_mask = ex_mask.astype(np.float32)
            label = label.astype(np.float32)
            ex_mask = np.where(ex_mask == 0.0, -10, ex_mask)

            result_scan = result_scan + ex_mask
            label = label + ex_mask

            if np.count_nonzero(label == 2.0) == 0:
                nan_num += 1
                cnt += 1
                continue
            cnt += 1

            fps, tpr = fp_per_scan(result_scan, label)

            fps_list.append(fps)
            sens_list.append(tpr)

        fps_itp = np.linspace(MIN_FROC, MAX_FROC, num=10001)
        sens_itp = None
        for list_i in range(len(fps_list)):
            fps_list[list_i] /= (all_scan_num - nan_num)
            sens_list[list_i] /= (all_scan_num - nan_num)
            if sens_itp is None:
                sens_itp = np.interp(fps_itp, fps_list[list_i], sens_list[list_i])
            else:
                sens_itp += np.interp(fps_itp, fps_list[list_i], sens_list[list_i])
        print(fps_itp)
        print(sens_itp)

    @property
    def model_dir(self):
        return "{}_{}".format(
            self.model_name, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
