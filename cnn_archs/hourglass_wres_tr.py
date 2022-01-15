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

def make_wts_and_bias(var_suffix, input_channels, output_channels, type):
    if type == 'normal':
        weights = tf.get_variable('W' + var_suffix, shape=(3, 3, input_channels, output_channels), initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('B' + var_suffix, shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    elif type == 'transpose':
        weights = tf.get_variable('W' + var_suffix, shape=(3, 3, output_channels, input_channels), initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable('B' + var_suffix, shape=(output_channels), initializer=tf.contrib.layers.xavier_initializer())
    return weights, biases

def residual_block(x, output_channels, scope_name):
    input_channels =  x.shape[3]
    with tf.variable_scope(scope_name, reuse= tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('1', input_channels, output_channels, 'normal')
    y1 = conv2d(x, weights, biases)

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('2', output_channels, output_channels, 'normal')
    y2 = conv2d(x, weights, biases)

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('3', output_channels, output_channels, 'normal')
    y3 = conv2d(x, weights, biases)

    y3 = y3 + x
    return y1, y2, y3


def conv_net(x, output_shape):

    # start
    with tf.variable_scope('start', reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('1', x.shape[3], 32, 'normal')
    conv0 = conv2d(x, weights, biases)

    # block1
    conv1, conv2, conv3 = residual_block(conv0, 32, 'block1')
    mp1 = maxpool2d(conv3, 2)

    # block2
    conv4, conv5, conv6 = residual_block(mp1, 32, 'block2')
    mp2 = maxpool2d(conv6, 2)

    # block3
    conv7, conv8, conv9 = residual_block(mp2, 32, 'block3')
    mp3 = maxpool2d(conv9, 2)

    # block4
    conv10, conv11, conv12 = residual_block(mp3, 32, 'block4')

    # sideblock1
    conv13, conv14, conv15 = residual_block(conv3, 32, 'sideblock1')

    # sideblock2
    conv16, conv17, conv18 = residual_block(conv6, 32, 'sideblock2')

    # sideblock3
    conv19, conv20, conv21 = residual_block(conv9, 32, 'sideblock3')

    # up1
    mp4 = maxpool2d(conv21, 2)
    conv12 = conv12 + mp4
    with tf.variable_scope('up_block1', reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('U1', 32, 32, 'transpose')
    conv22 = conv2d_transpose(conv12, weights, biases, [tf.shape(conv12)[0], 128, 128, 32], strides=2)
    conv23, conv24, conv25 = residual_block(conv22, 32, 'up_block1')

    # up2
    mp5 = maxpool2d(conv18, 2)
    conv25 = conv25 + mp5
    with tf.variable_scope('up_block2', reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('U1', 32, 32, 'transpose')
    conv26 = conv2d_transpose(conv25, weights, biases, [tf.shape(conv12)[0], 256, 256, 32], strides=2)
    conv27, conv28, conv29 = residual_block(conv26, 32, 'up_block2')

    # up3
    mp6 = maxpool2d(conv15, 2)
    conv29 = conv29 + mp6
    with tf.variable_scope('up_block3', reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('U1', 32, 32, 'transpose')
    conv30 = conv2d_transpose(conv29, weights, biases, [tf.shape(conv12)[0], 512, 512, 32], strides=2)
    conv31, conv32, conv33 = residual_block(conv30, 32, 'up_block3')

    # proj
    with tf.variable_scope('proj', reuse=tf.AUTO_REUSE) as scope:
        weights, biases = make_wts_and_bias('1', 32, output_shape, 'normal')
    conv34 = conv2d(conv33, weights, biases)
    out = conv34
    return out