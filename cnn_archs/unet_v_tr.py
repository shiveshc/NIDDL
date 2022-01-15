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

def make_wts_and_bias(var_suffix, input_channels, output_channels, type):
    if type == 'normal':
        weights = tf.get_variable('W' + var_suffix, shape=(3, 3, input_channels, output_channels), initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('B' + var_suffix, shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    elif type == 'transpose':
        weights = tf.get_variable('W' + var_suffix, shape=(3, 3, output_channels, input_channels), initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('B' + var_suffix, shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    return weights, biases

def conv_block(x, output_channels, scope_name):
    input_channels =  x.shape[3]
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('1', input_channels, output_channels, 'normal')
    y1 = conv2d(x, weights, biases)

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('2', output_channels, output_channels, 'normal')
    y2 = conv2d(y1, weights, biases)

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('3', output_channels, output_channels, 'normal')
    y3 = conv2d(y2, weights, biases)
    return y1, y2, y3


def conv_net(x, output_shape):

    ####### Unet 1 #########
    # block1
    conv1, conv2, conv3 = conv_block(x, 32, 'block1')

    # block2
    mp1 = maxpool2d(conv3, 2)
    conv4, conv5, conv6 = conv_block(mp1, 32, 'block2')

    # block3
    mp2 = maxpool2d(conv6, 2)
    conv7, conv8, conv9  = conv_block(mp2, 32, 'block3')

    # block4
    mp3 = maxpool2d(conv9, 2)
    conv10, conv11, conv12 = conv_block(mp3, 32, 'block4')

    # block5
    mp4 = maxpool2d(conv12, 2)
    conv13, conv14, conv15 = conv_block(mp4, 32, 'block5')

    # up block 1
    with tf.variable_scope('up_block1', reuse=tf.AUTO_REUSE) as scope:
        up_w, up_b = make_wts_and_bias('U1', 32, 32, 'transpose')
    uconv1 = conv2d_transpose(conv15, up_w, up_b, [tf.shape(conv3)[0], 64, 64, 32], strides=2)
    uconv1 = tf.concat([uconv1, conv12], axis=3)
    conv16, conv17, conv18 = conv_block(uconv1, 32, 'up_block1')

    # up block 2
    with tf.variable_scope('up_block2', reuse=tf.AUTO_REUSE) as scope:
        up_w, up_b = make_wts_and_bias('U1', 32, 32, 'transpose')
    uconv2 = conv2d_transpose(conv18, up_w, up_b, [tf.shape(conv3)[0], 128, 128, 32], strides=2)
    uconv2 = tf.concat([uconv2, conv9], axis=3)
    conv19, conv20, conv21 = conv_block(uconv2, 32, 'up_block2')

    # up block 3
    with tf.variable_scope('up_block3', reuse=tf.AUTO_REUSE) as scope:
        up_w, up_b = make_wts_and_bias('U1', 32, 32, 'transpose')
    uconv3 = conv2d_transpose(conv21, up_w, up_b, [tf.shape(conv3)[0], 256, 256, 32], strides=2)
    uconv3 = tf.concat([uconv3, conv6], axis=3)
    conv22, conv23, conv24 = conv_block(uconv3, 32, 'up_block3')

    # up block 4
    with tf.variable_scope('up_block4', reuse=tf.AUTO_REUSE) as scope:
        up_w, up_b = make_wts_and_bias('U1', 32, 32, 'transpose')
    uconv4 = conv2d_transpose(conv24, up_w, up_b, [tf.shape(conv3)[0], 512, 512, 32], strides=2)
    uconv4 = tf.concat([uconv4, conv3], axis=3)
    conv25, conv26, conv27 = conv_block(uconv4, 32, 'up_block4')

    # proj_to_img
    with tf.variable_scope('proj', reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('p', 32, output_shape, 'normal')
    conv28 = conv2d(conv27, weights, biases)
    out1 = conv28

    return out1