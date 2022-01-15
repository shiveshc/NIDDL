import tensorflow as tf

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv2d_transpose(x, W, b, output, strides=2):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d_transpose(x, W, output_shape= output, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv_net(x, weights, biases):
    # block1
    conv1 = conv2d(x, weights['wc1'], biases['b1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['b2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['b3'])

    # block2
    mp1 = maxpool2d(conv3, 2)
    conv4 = conv2d(mp1, weights['wc4'], biases['b4'])
    conv5 = conv2d(conv4, weights['wc5'], biases['b5'])
    conv6 = conv2d(conv5, weights['wc6'], biases['b6'])

    # block3
    mp2 = maxpool2d(conv6, 2)
    conv7 = conv2d(mp2, weights['wc7'], biases['b7'])
    conv8 = conv2d(conv7, weights['wc8'], biases['b8'])
    conv9 = conv2d(conv8, weights['wc9'], biases['b9'])

    # block4
    mp3 = maxpool2d(conv9, 2)
    conv10 = conv2d(mp3, weights['wc10'], biases['b10'])
    conv11 = conv2d(conv10, weights['wc11'], biases['b11'])
    conv12 = conv2d(conv11, weights['wc12'], biases['b12'])

    # up block 1
    uconv1 = conv2d_transpose(conv12, weights['wuc1'], biases['ub1'], [tf.shape(conv3)[0], 215, 163, 128], strides=2)

    # up block 2
    uconv2 = conv2d_transpose(uconv1, weights['wuc2'], biases['ub2'], [tf.shape(conv3)[0], 430, 325, 128], strides=2)

    # up block 3
    uconv3 = conv2d_transpose(uconv2, weights['wuc3'], biases['ub3'], [tf.shape(conv3)[0], 860, 650, 128], strides=2)

    conv22 = conv2d(uconv3, weights['wc22'], biases['b22'])
    out = conv22
    return out


def wts_and_bias():
    weights = {'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc2': tf.get_variable('W1', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc3': tf.get_variable('W2', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc4': tf.get_variable('W3', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
               'wc5': tf.get_variable('W4', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
               'wc6': tf.get_variable('W5', shape=(3, 3, 64, 64), initializer=tf.contrib.layers.xavier_initializer()),
               'wc7': tf.get_variable('W6', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
               'wc8': tf.get_variable('W7', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
               'wc9': tf.get_variable('W8', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
               'wc10': tf.get_variable('W9', shape=(3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer()),
               'wc11': tf.get_variable('W10', shape=(3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer()),
               'wc12': tf.get_variable('W11', shape=(3, 3, 256, 256), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc1': tf.get_variable('W12', shape=(3, 3, 128, 256), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc2': tf.get_variable('W16', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc3': tf.get_variable('W20', shape=(3, 3, 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
               'wc22': tf.get_variable('W24', shape=(1, 1, 128, 1), initializer=tf.contrib.layers.xavier_initializer())}

    biases = {'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b3': tf.get_variable('B2', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b4': tf.get_variable('B3', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
              'b5': tf.get_variable('B4', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
              'b6': tf.get_variable('B5', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
              'b7': tf.get_variable('B6', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
              'b8': tf.get_variable('B7', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
              'b9': tf.get_variable('B8', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
              'b10': tf.get_variable('B9', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
              'b11': tf.get_variable('B10', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
              'b12': tf.get_variable('B11', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
              'ub1': tf.get_variable('B12', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
              'ub2': tf.get_variable('B16', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
              'ub3': tf.get_variable('B20', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
              'b22': tf.get_variable('B24', shape=(1), initializer=tf.contrib.layers.xavier_initializer()),}

    return weights, biases