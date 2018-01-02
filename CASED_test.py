import time, pickle
from ops import *
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_blocks as patch_blocks
from math import ceil
import numpy as np
class CASED(object) :
    def __init__(self, sess, batch_size, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = 'LUNA16'
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.model_name = "CASED"     # name for checkpoint

        self.c_dim = 1
        self.y_dim = 2 # nodule ? or non_nodule ?
        self.block_size = 72

    def cased_network(self, x, reuse=False, scope='CASED_NETWORK'):
        with tf.variable_scope(scope, reuse=reuse) :
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

            x = deconv_layer(x, channels=128, kernel=4, stride=2,layer_name='deconv1')
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

        bs = self.batch_size
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

        validation_sub_n = 0
        subset_name = 'subset' + str(validation_sub_n)
        image_paths = glob.glob("/data/jhkim/LUNA16/original/subset" + str(validation_sub_n) + '/*.mhd')
        all_scan_num = len(image_paths)
        sens_list = None
        nan_num = 0
        for scan in image_paths :

            scan_name = os.path.split(scan)[1].replace('.mhd', '')
            scan_npy = '/data2/jhkim/npydata/' + subset_name + '/' + scan_name + '.npy'
            label_npy = '/data2/jhkim/npydata/' + subset_name + '/' + scan_name + '.label.npy'

            image = np.transpose(np.load(scan_npy))
            label = np.transpose(np.load(label_npy))

            if np.count_nonzero(label) == 0 :
                nan_num += 1
                continue

            print(np.shape(image))
            print(np.shape(label))

            pad_list = []
            for i in range(3) :
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

            # np.min padding
            image = np.pad(image, pad_width=[ [pad_list[0], pad_list[1]], [pad_list[2], pad_list[3]], [pad_list[4], pad_list[5]] ], mode='constant', constant_values=np.min(image))

            label = np.pad(label, pad_width=[ [pad_list[0], pad_list[1]], [pad_list[2], pad_list[3]], [pad_list[4], pad_list[5]] ], mode='constant', constant_values=np.min(label))

            print('padding !')
            print(np.shape(image))
            print(np.shape(label))

            image_blocks = patch_blocks(image, block_shape=(block_size, block_size, block_size))
            len_x = len(image_blocks)
            len_y = len(image_blocks[0])
            len_z = len(image_blocks[0, 0])

            result_scan = None
            for x_i in range(len_x):
                x = None
                for y_i in range(len_y):
                    y = None
                    for z_i in range(len_z):
                        scan = np.expand_dims(np.expand_dims(image_blocks[x_i, y_i, z_i], axis=-1), axis=0) # 1 72 72 72 1
                        scan = np.pad(scan, pad_width=[[0, 0], [30, 30], [30, 30], [30, 30], [0, 0]],  mode='constant', constant_values=np.min(scan))
                        test_feed_dict = {
                            self.inputs: scan
                        }

                        logits = self.sess.run(
                            self.softmax_logits, feed_dict=test_feed_dict
                        ) # [1, 72, 72, 72, 2]
                        logits = np.squeeze(logits, axis=0) # [72,72,72,2]
                        # logits = np.squeeze(np.argmax(logits, axis=-1), axis=0)
                        """
                        [1, 72, 72, 72, 2] -> [1, 72, 72, 72] -> [72,72,72]
                        """

                        y = logits if y is None else np.concatenate((y, logits), axis=2)

                    x = y if x is None else np.concatenate((x, y), axis=1)

                result_scan = x if result_scan is None else np.concatenate((result_scan, x), axis=0)
            # print(result) # 3d original size

            with open('include.pkl', 'rb') as f:
                coords_dict = pickle.load(f, encoding='bytes')

            with open('exclude.pkl', 'rb') as f:
                exclude_dict = pickle.load(f, encoding='bytes')

            exclude_coords = exclude_dict[scan_name]

            for ex in exclude_coords :
                ex[0] = ex[0] + (pad_list[0] + pad_list[1]) // 2
                ex[1] = ex[1] + (pad_list[2] + pad_list[3]) // 2
                ex[2] = ex[2] + (pad_list[4] + pad_list[5]) // 2
                ex_diameter = ex[3]
                if ex_diameter < 0.0 :
                    ex_diameter = 10.0
                exclude_position = (ex[0], ex[1], ex[2])
                exclude_mask = create_exclude_mask(result_scan.shape, exclude_position, ex_diameter )
                exclude_mask = np.expand_dims(exclude_mask, axis=-1)
                exclude_mask = indices_to_one_hot(exclude_mask, 2)

                result_scan = np.logical_and(result_scan, exclude_mask) # [72, 72, 72, 2]
            """
            coords = coords_dict[scan_name]
            cnt = 0
            for c in coords :
                print('******** result ********')
                print(np.shape(result_scan))
                print(np.shape(label))
                print(c)
                x_coords = c[0] + (pad_list[0] + pad_list[1]) // 2
                y_coords = c[1] + (pad_list[2] + pad_list[3]) // 2
                z_coords = c[2] + (pad_list[4] + pad_list[5]) // 2
                offset = 34
                result_scan_img = result_scan[int(x_coords - offset): int(x_coords + offset), int(y_coords - offset):int(y_coords + offset), z_coords]
                label_scan = label[int(x_coords - offset): int(x_coords + offset), int(y_coords - offset):int(y_coords + offset), z_coords]

                plt.imsave('./image/test_{}_{}.png'.format(scan_num,cnt), result_scan_img, cmap=plt.cm.gray)
                plt.imsave('./image/label_{}_{}.png'.format(scan_num,cnt), label_scan, cmap=plt.cm.gray)
                cnt += 1
                scan_num += 1
            """

            label = indices_to_one_hot(label, 2)

            if sens_list is None :
                sens_list = fp_per_scan(result_scan, label)
            else :
                sens_list += fp_per_scan(result_scan, label)

        fp_list = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        sens_list /= (all_scan_num - nan_num)

        for i in range(len(fp_list)) :
            print('{} : {}'.format(fp_list[i], sens_list[i]))

        print('Average sensitivity : {}'.format(np.mean(sens_list)))





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
