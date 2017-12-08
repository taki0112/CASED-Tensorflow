import time
from heapq import nlargest
from random import uniform

from ops import *
from utils import *
class CASED(object) :
    def __init__(self, sess, epoch, batch_size, test_batch_size, num_gpu, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = 'LUNA16'
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.predictor_batch_size = batch_size * 2
        self.model_name = "CASED"     # name for checkpoint
        self.num_gpu = num_gpu
        self.total_subset = 10

        self.x = 68
        self.y = 68
        self.z = 68
        self.c_dim = 1
        self.y_dim = 2 # nodule ? or non_nodule ?
        self.out_dim = 8

        self.weight_decay = 1e-4
        self.lr_decay = 1.0  # not decay
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.M = 10

    def cased_network(self, x, reuse=False, scope='CASED_NETWORK'):
        with tf.variable_scope(scope, reuse=reuse) :
            # print(np.shape(x))
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

            x = deconv_layer(x, channels=256, kernel=4, stride=2, layer_name='deconv1')
            x = copy_crop(crop_layer=up_conv3, in_layer=x, crop=5)

            x = conv_layer(x, channels=128, kernel=1, stride=1, layer_name='conv6')
            x = conv_layer(x, channels=64, kernel=1, stride=1, layer_name='conv7')

            x = deconv_layer(x, channels=128, kernel=4, stride=2, layer_name='deconv2')
            x = copy_crop(crop_layer=up_conv2, in_layer=x, crop=7)

            x = conv_layer(x, channels=64, kernel=1, stride=1, layer_name='conv8')
            x = conv_layer(x, channels=32, kernel=1, stride=1, layer_name='conv9')

            x = deconv_layer(x, channels=64, kernel=4, stride=2, layer_name='deconv3')
            x = copy_crop(crop_layer=up_conv1, in_layer=x, crop=8)

            x = conv_layer(x, channels=32, kernel=1, stride=1, layer_name='conv10')
            x = conv_layer(x, channels=32, kernel=1, stride=1, layer_name='conv11')

            logits = conv_layer(x, channels=2, kernel=1, stride=1, activation=None, layer_name='conv12')
            x = softmax(logits)

            return logits, x

    def build_model(self):
        patch_dims = [self.x, self.y, self.z, self.c_dim]
        bs = self.batch_size
        p_bs = self.predictor_batch_size * self.num_gpu
        y_dims = [self.out_dim, self.out_dim, self.out_dim, self.y_dim]
        self.decay_lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + patch_dims, name='patch')
        self.p_inputs = tf.placeholder(tf.float32, [p_bs] + patch_dims, name='p_patch')

        # labels
        self.y = tf.placeholder(tf.float32, [bs] + y_dims, name='y')  # for loss
        self.p_y = tf.placeholder(tf.float32, [p_bs] + y_dims, name='p_y')

        self.logits, self.softmax_logits = self.cased_network(self.inputs)
        self.P_logits, self.P_softmax_logits = self.cased_network(self.inputs, reuse=True)

        """ Predictor Loss """
        self.x_dict = dict()
        for gpu_i in range(self.num_gpu) :
            with tf.device('/gpu:%d' % gpu_i) :
                index_start = self.predictor_batch_size * gpu_i
                index_end = self.predictor_batch_size * (gpu_i + 1)
                self.predictor_logits, _ = self.cased_network(self.p_inputs[index_start : index_end], reuse=True)
                self.P_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.p_y[index_start : index_end],
                                                                                     logits=self.predictor_logits), axis=[1, 2, 3])
                self.x_dict.update(
                    {index_start + i: self.P_loss[i] for i in range(self.predictor_batch_size)}
                )

        """ Loss function """
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss = self.loss + self.l2_loss*self.weight_decay

        self.correct_prediction = tf.equal(tf.argmax(self.softmax_logits, -1), tf.argmax(self.y, -1))
        self.P_correct_prediction = tf.equal(tf.argmax(self.P_softmax_logits, -1), tf.argmax(self.y, -1))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.P_accuracy = tf.reduce_mean(tf.cast(self.P_correct_prediction, tf.float32))

        self.sensitivity = sensitivity(labels=self.y, logits=self.softmax_logits)
        self.P_sensitivity = sensitivity(labels=self.y, logits=self.P_softmax_logits)

        self.optim = tf.train.MomentumOptimizer(learning_rate=self.decay_lr, momentum=self.momentum, use_nesterov=True).minimize(self.loss)

        """ Summary """
        c_lr = tf.summary.scalar('cosine_lr', self.decay_lr)
        c_loss = tf.summary.scalar('loss', self.loss)
        c_acc = tf.summary.scalar('acc', self.accuracy)
        c_recall = tf.summary.scalar('sensitivity', self.sensitivity)
        self.c_sum = tf.summary.merge([c_lr, c_loss, c_acc, c_recall])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.total_subset)
            start_batch_id = checkpoint_counter - start_epoch * self.total_subset
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 0
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        count_temp = 0
        # 10 counter = 1 epoch
        for epoch in range(start_epoch, self.epoch):
            for sub_n in range(start_batch_id, self.total_subset) :
                train_acc = 0.0
                train_recall = 0.0
                nan_num = 0
                prob = 1.0
                train_lr = self.learning_rate
                nodule_patch, all_patch, nodule_y, all_y = prepare_date(sub_n)
                M = len(all_patch)
                num_batches = M // self.batch_size
                print('finish prepare data : ', M)
                for idx in range(num_batches):
                    if idx == int((num_batches * 0.5)) :
                         train_lr = train_lr * self.lr_decay
                         print("*** now learning rate : {} ***\n".format(train_lr))

                    each_time = time.time()
                    p = uniform(0,1)
                    print('probability M : ', prob)
                    print('condition P : ', p)
                    if p <= prob :
                        random_index = np.random.choice(len(nodule_patch), size=self.batch_size, replace=False)
                        batch_patch = nodule_patch[random_index]
                        batch_y = nodule_y[random_index]
                    else :
                        predict_batch = self.predictor_batch_size * self.num_gpu
                        total_predict_index = M // predict_batch
                        predict_dict = dict()
                        for p_idx in range(total_predict_index) :
                            result = dict()
                            batch_patch = all_patch[predict_batch*p_idx : predict_batch*(p_idx+1)]
                            batch_y = all_y[predict_batch*p_idx : predict_batch*(p_idx+1)]
                            predictor_feed_dict = {
                                self.p_inputs: batch_patch, self.p_y : batch_y
                            }
                            temp_x = self.sess.run(self.x_dict, feed_dict=predictor_feed_dict)

                            if p_idx != 0 :
                                for k,v in temp_x.items() :
                                    new_k = k + predict_batch * p_idx
                                    result[new_k] = v
                            else :
                                for k,v in temp_x.items() :
                                    result[k] = v

                            predict_dict.update(result)
                            index = nlargest(self.batch_size, predict_dict, key=predict_dict.get)
                            predict_dict = {s_idx: predict_dict[s_idx] for s_idx in index}

                        g_r_index = list(predict_dict.keys())
                        batch_patch = all_patch[g_r_index]
                        batch_y = all_y[g_r_index]

                    # import ipdb; ipdb.set_trace()
                    prob *= pow(1/M, 1/num_batches)
                    train_feed_dict = {
                        self.inputs: batch_patch, self.y : batch_y,
                        self.decay_lr : train_lr
                    }

                    _, summary_str_c, c_loss, c_acc, c_recall = self.sess.run(
                        [self.optim, self.c_sum, self.loss, self.accuracy, self.sensitivity],
                        feed_dict=train_feed_dict)
                    self.writer.add_summary(summary_str_c, count_temp)
                    count_temp += 1

                    if np.isnan(c_recall) :
                        train_acc += c_acc
                        # train_recall += 0
                        nan_num += 1
                    else :
                        train_acc += c_acc
                        train_recall += c_recall
                    # display training status
                    print("Epoch: [%2d], Sub_n: [%2d], [%4d/%4d] time: %4.4f, each_time: %4.4f, c_loss: %.8f, c_acc: %.4f, c_recall: %.4f" \
                          % (epoch, sub_n, idx, num_batches, time.time() - start_time, time.time() - each_time, c_loss, c_acc, c_recall))

                train_acc /= num_batches
                train_recall /= (num_batches - nan_num)

                summary_train = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy', simple_value=train_acc),
                                                 tf.Summary.Value(tag='train_recall', simple_value=train_recall)])
                self.writer.add_summary(summary_train, counter)

                line = "Epoch: [%2d], Sub_n: [%2d], train_acc: %.4f, train_recall: %.4f\n" % (epoch, sub_n, train_acc, train_recall)
                print(line)
                with open(os.path.join(self.result_dir, 'train_logs.txt'), 'a') as f:
                    f.write(line)
                # save model
                self.save(self.checkpoint_dir, counter)
                counter += 1
                del nodule_patch
                del nodule_y
                del all_patch
                del all_y

            total_test_acc = 0.0
            total_test_recall = 0.0

            for test_sub_n in range(self.total_subset) :
                # classifier test
                all_patch, all_y = test_data(test_sub_n)
                test_acc = 0.0
                test_recall = 0.0
                nan_num = 0
                test_idx = len(all_patch) // self.test_batch_size

                for idx in range(test_idx) :
                    batch_patch = all_patch[idx * self.test_batch_size : (idx+1) * self.test_batch_size]
                    batch_y = all_y[idx * self.test_batch_size : (idx+1) * self.test_batch_size]

                    test_feed_dict = {
                        self.inputs : batch_patch, self.y : batch_y
                    }

                    acc_, recall_ = self.sess.run([self.P_accuracy, self.P_sensitivity], feed_dict=test_feed_dict)

                    if np.isnan(recall_) :
                        test_acc += acc_
                        nan_num += 1
                    else :
                        test_acc += acc_
                        test_recall += recall_
                test_acc /= test_idx
                test_recall /= (test_idx - nan_num)

                total_test_acc += test_acc
                total_test_recall += test_recall

                del all_patch
                del all_y
                line = "::: sub test finish ::: test_sub_n: [%2d], sub_test_acc: %.4f, sub_test_recall: %.4f\n" % (test_sub_n, test_acc, test_recall)
                print(line)
                with open(os.path.join(self.result_dir, 'test_logs.txt'), 'a') as f:
                    f.write(line)

            total_test_acc /= self.total_subset
            total_test_recall /= self.total_subset

            summary_test = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=total_test_acc),
                                             tf.Summary.Value(tag='test_recall', simple_value=total_test_recall)])
            self.writer.add_summary(summary_test, epoch)

            line = "Epoch: [%2d], test_acc: %.4f, test_recall: %.4f\n\n" % (epoch, total_test_acc, total_test_recall)
            print(line)
            with open(os.path.join(self.result_dir,'test_logs.txt'), 'a') as f:
                f.write(line)
            start_batch_id = 0
            # save model for final step
            self.save(self.checkpoint_dir, counter)
            counter += 1

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size,)

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
