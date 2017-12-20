import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as he_init
from tensorflow.contrib.layers import l2_regularizer

def conv_layer(x, channels, kernel=3, stride=1, activation='relu', padding='VALID', layer_name='_conv3d') :
    with tf.name_scope(layer_name) :
        if activation == None :
            return tf.layers.conv3d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(),
                                    strides=stride, padding=padding)
        else :
            return tf.layers.conv3d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(),
                                    strides=stride, padding=padding, activation=relu)

def deconv_layer(x, channels, kernel=4, stride=2, padding='SAME', layer_name='_deconv3d') :
    with tf.name_scope(layer_name) :
        x = tf.layers.conv3d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(),
                                       strides=stride, padding=padding, use_bias=False)
        return x
def flatten(x) :
    return tf.layers.flatten(x)

def fully_connected(x, unit=2, layer_name='fully') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(x, units=unit)

def max_pooling(x, kernel=2, stride=2, padding='VALID') :
    x = tf.layers.max_pooling3d(inputs=x, pool_size=kernel, strides=stride, padding=padding)
    x = tf.concat([x, x], axis=-1)
    return x


def copy_crop(crop_layer, in_layer):
    crop = []
    for i in range(1, 4):
        crop_left = (tf.shape(crop_layer)[i] - tf.shape(in_layer)[i]) // 2
        crop.append(crop_left)

        crop_right = tf.cond(tf.equal(tf.shape(crop_layer)[i] - crop_left*2, tf.shape(in_layer)[i]) ,
                       lambda : crop_left,
                       lambda : crop_left + 1)
        crop.append(crop_right)

    crop_layer = crop_layer[:, crop[0]: -crop[1], crop[2]: -crop[3], crop[4]: -crop[5], :]

    return tf.concat([crop_layer, in_layer], axis=-1)

def relu(x) :
    return tf.nn.relu(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def softmax(x) :
    return tf.nn.softmax(x)
