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

def make_wts_and_bias(weights, biases, input_channels, output_channels, type):
    curr_weights_num = len(weights)
    curr_biases_num = len(biases)
    if type == 'normal':
        new_w = 'wc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(3, 3, input_channels, output_channels), initializer=tf.contrib.layers.xavier_initializer())
        new_b = 'b' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    elif type == 'transpose':
        new_w = 'wuc' + str(curr_weights_num + 1)
        weights[new_w] = tf.get_variable('W' + str(curr_weights_num + 1), shape=(3, 3, output_channels, input_channels), initializer=tf.contrib.layers.xavier_initializer())
        new_b = 'ub' + str(curr_biases_num + 1)
        biases[new_b] = tf.get_variable('B' + str(curr_biases_num + 1), shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    return weights, biases, new_w, new_b

def residual_block(x, weights, biases, output_channels):
    input_channels =  x.shape[3]
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, input_channels, output_channels, 'normal')
    y1 = conv2d(x, weights[new_w], biases[new_b])

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, output_channels, output_channels, 'normal')
    y2 = conv2d(y1, weights[new_w], biases[new_b])

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, output_channels, output_channels, 'normal')
    y3 = conv2d(y2, weights[new_w], biases[new_b])

    y3 = y3 + x
    return y1, y2, y3, weights, biases


def conv_net(x, output_shape):
    weights = {}
    biases = {}
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, x.shape[3], 32, 'normal')
    conv0 = conv2d(x, weights[new_w], biases[new_b])

    # block1
    conv1, conv2, conv3, weights, biases = residual_block(conv0, weights, biases, 32)
    mp1 = maxpool2d(conv3, 2)

    # block2
    conv4, conv5, conv6, weights, biases = residual_block(mp1, weights, biases, 32)
    mp2 = maxpool2d(conv6, 2)

    # block3
    conv7, conv8, conv9, weights, biases = residual_block(mp2, weights, biases, 32)
    mp3 = maxpool2d(conv9, 2)

    # block4
    conv10, conv11, conv12, weights, biases = residual_block(mp3, weights, biases, 32)

    # sideblock1
    conv13, conv14, conv15, weights, biases = residual_block(conv3, weights, biases, 32)

    # sideblock2
    conv16, conv17, conv18, weights, biases = residual_block(conv6, weights, biases, 32)

    # sideblock3
    conv19, conv20, conv21, weights, biases = residual_block(conv9, weights, biases, 32)

    # up1
    mp4 = maxpool2d(conv21, 2)
    conv12 = conv12 + mp4
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv22 = conv2d_transpose(conv12, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 128, 128, 32], strides=2)
    conv23, conv24, conv25, weights, biases = residual_block(conv22, weights, biases, 32)

    # up2
    mp5 = maxpool2d(conv18, 2)
    conv25 = conv25 + mp5
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv26 = conv2d_transpose(conv25, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 256, 256, 32], strides=2)
    conv27, conv28, conv29, weights, biases = residual_block(conv26, weights, biases, 32)

    # up3
    mp6 = maxpool2d(conv15, 2)
    conv29 = conv29 + mp6
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv30 = conv2d_transpose(conv29, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 512, 512, 32], strides=2)
    conv31, conv32, conv33, weights, biases = residual_block(conv30, weights, biases, 32)

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, output_shape, 'normal')
    conv34 = conv2d(conv33, weights[new_w], biases[new_b])
    out = conv34
    return out