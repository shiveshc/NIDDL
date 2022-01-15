import tensorflow as tf

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv2d_transpose(x, W, b, output, strides=2):
    # Conv2D transpose wrapper, with bias and relu activation
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

def conv_block(x, weights, biases, output_channels):
    input_channels =  x.shape[3]
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, input_channels, output_channels, 'normal')
    y1 = conv2d(x, weights[new_w], biases[new_b])

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, output_channels, output_channels, 'normal')
    y2 = conv2d(y1, weights[new_w], biases[new_b])

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, output_channels, output_channels, 'normal')
    y3 = conv2d(y2, weights[new_w], biases[new_b])
    return y1, y2, y3, weights, biases


def conv_net(x):
    weights = {}
    biases = {}

    ####### Unet 1 #########
    # block1
    conv1, conv2, conv3, weights, biases = conv_block(x, weights, biases, 32)

    # block2
    mp1 = maxpool2d(conv3, 2)
    conv4, conv5, conv6, weights, biases = conv_block(mp1, weights, biases, 32)

    # block3
    mp2 = maxpool2d(conv6, 2)
    conv7, conv8, conv9, weights, biases = conv_block(mp2, weights, biases, 32)

    # block4
    mp3 = maxpool2d(conv9, 2)
    conv10, conv11, conv12, weights, biases = conv_block(mp3, weights, biases, 32)

    # block5
    mp4 = maxpool2d(conv12, 2)
    conv13, conv14, conv15, weights, biases = conv_block(mp4, weights, biases, 32)

    # up block 1
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv1 = conv2d_transpose(conv15, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 108, 82, 32], strides=2)
    uconv1 = tf.concat([uconv1, conv12], axis=3)
    conv16, conv17, conv18, weights, biases = conv_block(uconv1, weights, biases, 32)

    # up block 2
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv2 = conv2d_transpose(conv18, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 215, 163, 32], strides=2)
    uconv2 = tf.concat([uconv2, conv9], axis=3)
    conv19, conv20, conv21, weights, biases = conv_block(uconv2, weights, biases, 32)

    # up block 3
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv3 = conv2d_transpose(conv21, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 430, 325, 32], strides=2)
    uconv3 = tf.concat([uconv3, conv6], axis=3)
    conv22, conv23, conv24, weights, biases = conv_block(uconv3, weights, biases, 32)

    # up block 4
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv4 = conv2d_transpose(conv24, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 860, 650, 32], strides=2)
    uconv4 = tf.concat([uconv4, conv3], axis=3)
    conv25, conv26, conv27, weights, biases = conv_block(uconv4, weights, biases, 32)

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 1, 'normal')
    conv28 = conv2d(conv27, weights[new_w], biases[new_b])
    out1 = conv28

    ####### Unet 2 #######
    conv27 = tf.concat([conv27, x], axis=3)

    # block1
    conv29, conv30, conv31, weights, biases = conv_block(conv27, weights, biases, 32)

    # block2
    mp5 = maxpool2d(conv31, 2)
    conv32, conv33, conv34, weights, biases = conv_block(mp5, weights, biases, 32)

    # block3
    mp6 = maxpool2d(conv34, 2)
    conv35, conv36, conv37, weights, biases = conv_block(mp6, weights, biases, 32)

    # block4
    mp7 = maxpool2d(conv37, 2)
    conv38, conv39, conv40, weights, biases = conv_block(mp7, weights, biases, 32)

    # block5
    mp8 = maxpool2d(conv40, 2)
    conv41, conv42, conv43, weights, biases = conv_block(mp8, weights, biases, 32)

    # up block 1
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv5 = conv2d_transpose(conv43, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 108, 82, 32], strides=2)
    uconv5 = tf.concat([uconv5, conv40], axis=3)
    conv44, conv45, conv46, weights, biases = conv_block(uconv5, weights, biases, 32)

    # up block 2
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv6 = conv2d_transpose(conv46, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 215, 163, 32], strides=2)
    uconv6 = tf.concat([uconv6, conv37], axis=3)
    conv47, conv48, conv49, weights, biases = conv_block(uconv6, weights, biases, 32)

    # up block 3
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv7 = conv2d_transpose(conv49, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 430, 325, 32], strides=2)
    uconv7 = tf.concat([uconv7, conv34], axis=3)
    conv50, conv51, conv52, weights, biases = conv_block(uconv7, weights, biases, 32)

    # up block 4
    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 32, 'transpose')
    uconv8 = conv2d_transpose(conv52, weights[new_w], biases[new_b], [tf.shape(conv3)[0], 860, 650, 32], strides=2)
    uconv8 = tf.concat([uconv8, conv31], axis=3)
    conv53, conv54, conv55, weights, biases = conv_block(uconv8, weights, biases, 32)

    weights, biases, new_w, new_b = make_wts_and_bias(weights, biases, 32, 1, 'normal')
    conv56 = conv2d(conv55, weights[new_w], biases[new_b])
    out2 = conv56

    return out1, out2