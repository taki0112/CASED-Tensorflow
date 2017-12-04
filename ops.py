import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer as he_init
def conv_layer(x, channels, kernel=3, stride=1, activation='relu', padding='VALID', layer_name='_conv3d') :
    with tf.name_scope(layer_name) :
        if activation == None :
            return tf.layers.conv3d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(), strides=stride, padding=padding)
        else :
            return tf.layers.conv3d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(), strides=stride, padding=padding, activation=relu)

def deconv_layer(x, channels, kernel=4, stride=2, padding='VALID', layer_name='_deconv3d') :
    with tf.name_scope(layer_name) :
        x = tf.layers.conv3d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(), strides=stride, padding=padding)
        x = x[:, 1:-1, 1:-1, 1:-1, :]
        return x
def flatten(x) :
    return tf.layers.flatten(x)

def fully_connected(x, unit=2, layer_name='fully') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(x, units=unit, kernel_initializer=he_init())

def max_pooling(x, kernel=2, stride=2, padding='VALID') :
    x = tf.concat([x, x], axis=-1)
    x = tf.layers.max_pooling3d(inputs=x, pool_size=kernel, strides=stride, padding=padding)
    return x

def copy_crop(crop_layer, in_layer, crop) :
    x = int(crop_layer.shape[1]) # 1 or 2 or 3, 즉 x,y,z
    cropping_pixel = int((x - (x/crop)) * 0.5)
    crop_layer = crop_layer[:, cropping_pixel : -cropping_pixel, cropping_pixel : -cropping_pixel, cropping_pixel : -cropping_pixel, :]
    return tf.concat([crop_layer, in_layer], axis=-1)

def relu(x) :
    return tf.nn.relu(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def softmax(x) :
    return tf.nn.softmax(x)
