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


def conv_net(x):
    weights = {}
    biases = {}
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 1, 32, 'normal')
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
    mp4 = maxpool2d(conv12, 2)

    # block5
    conv13, conv14, conv15, weights, biases = residual_block(mp4, weights, biases, 32)
    mp5 = maxpool2d(conv15, 2)

    # block6
    conv16, conv17, conv18, weights, biases = residual_block(mp5, weights, biases, 32)

    # sideblock1
    conv19, conv20, conv21, weights, biases = residual_block(conv3, weights, biases, 32)

    # sideblock2
    conv22, conv23, conv24, weights, biases = residual_block(conv6, weights, biases, 32)

    # sideblock3
    conv25, conv26, conv27, weights, biases = residual_block(conv9, weights, biases, 32)

    # sideblock4
    conv28, conv29, conv30, weights, biases = residual_block(conv12, weights, biases, 32)

    # sideblock5
    conv31, conv32, conv33, weights, biases = residual_block(conv15, weights, biases, 32)

    # up1
    mp6 = maxpool2d(conv33, 2)
    conv18 = conv18 + mp6
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv34 = conv2d_transpose(conv18, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 54, 41, 32], strides=2)
    conv35, conv36, conv37, weights, biases = residual_block(conv34, weights, biases, 32)

    # up2
    mp7 = maxpool2d(conv30, 2)
    conv37 = conv37 + mp7
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv38 = conv2d_transpose(conv37, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 108, 82, 32], strides=2)
    conv39, conv40, conv41, weights, biases = residual_block(conv38, weights, biases, 32)

    # up3
    mp8 = maxpool2d(conv27, 2)
    conv41 = conv41 + mp8
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv42 = conv2d_transpose(conv41, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 215, 163, 32], strides=2)
    conv43, conv44, conv45, weights, biases = residual_block(conv42, weights, biases, 32)

    # up4
    mp9 = maxpool2d(conv24, 2)
    conv45 = conv45 + mp9
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv46 = conv2d_transpose(conv45, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 430, 325, 32], strides=2)
    conv47, conv48, conv49, weights, biases = residual_block(conv46, weights, biases, 32)

    # up5
    mp10 = maxpool2d(conv21, 2)
    conv49 = conv49 + mp10
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    conv50 = conv2d_transpose(conv49, weights[new_w], biases[new_b], [tf.shape(conv12)[0], 860, 650, 32], strides=2)
    conv51, conv52, conv53, weights, biases = residual_block(conv50, weights, biases, 32)

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 1, 'normal')
    conv54 = conv2d(conv53, weights[new_w], biases[new_b])
    out = conv54
    return out