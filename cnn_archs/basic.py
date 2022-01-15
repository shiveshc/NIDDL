import tensorflow as tf

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['b1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['b2'])
    conv3 = conv2d(conv2, weights['wc3'], biases['b3'])
    conv4 = conv2d(conv3, weights['wc4'], biases['b4'])

    out = conv4
    return out


def wts_and_bias():
    weights = {'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc2': tf.get_variable('W1', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc3': tf.get_variable('W2', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc4': tf.get_variable('W3', shape=(3, 3, 32, 1), initializer=tf.contrib.layers.xavier_initializer())}

    biases = {'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b3': tf.get_variable('B2', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b4': tf.get_variable('B3', shape=(1), initializer=tf.contrib.layers.xavier_initializer())}

    return weights, biases