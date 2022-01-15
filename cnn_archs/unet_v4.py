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

    # block5
    mp4 = maxpool2d(conv12, 2)
    conv13 = conv2d(mp4, weights['wc13'], biases['b13'])
    conv14 = conv2d(conv13, weights['wc14'], biases['b14'])
    conv15 = conv2d(conv14, weights['wc15'], biases['b15'])

    # block6
    mp5 = maxpool2d(conv15, 2)
    conv16 = conv2d(mp5, weights['wc16'], biases['b16'])
    conv17 = conv2d(conv16, weights['wc17'], biases['b17'])
    conv18 = conv2d(conv17, weights['wc18'], biases['b18'])

    # up block 1
    uconv1 = conv2d_transpose(conv18, weights['wuc1'], biases['ub1'], [tf.shape(conv3)[0], 54, 41, 32], strides=2)
    # uconv1 = conv2d(uconv1, weights['wrc1'], biases['rcb1'])
    uconv1 = tf.concat([uconv1, conv15], axis=3)
    conv19 = conv2d(uconv1, weights['wc19'], biases['b19'])
    conv20 = conv2d(conv19, weights['wc20'], biases['b20'])
    conv21 = conv2d(conv20, weights['wc21'], biases['b21'])

    # up block 2
    uconv2 = conv2d_transpose(conv21, weights['wuc2'], biases['ub2'], [tf.shape(conv3)[0], 108, 82, 32], strides=2)
    # uconv2 = conv2d(uconv2, weights['wrc2'], biases['rcb2'])
    uconv2 = tf.concat([uconv2, conv12], axis=3)
    conv22 = conv2d(uconv2, weights['wc22'], biases['b22'])
    conv23 = conv2d(conv22, weights['wc23'], biases['b23'])
    conv24 = conv2d(conv23, weights['wc24'], biases['b24'])

    # up block 3
    uconv3 = conv2d_transpose(conv24, weights['wuc3'], biases['ub3'], [tf.shape(conv3)[0], 215, 163, 32], strides=2)
    # uconv3 = conv2d(uconv3, weights['wrc3'], biases['rcb3'])
    uconv3 = tf.concat([uconv3, conv9], axis=3)
    conv25 = conv2d(uconv3, weights['wc25'], biases['b25'])
    conv26 = conv2d(conv25, weights['wc26'], biases['b26'])
    conv27 = conv2d(conv26, weights['wc27'], biases['b27'])

    # up block 4
    uconv4 = conv2d_transpose(conv27, weights['wuc4'], biases['ub4'], [tf.shape(conv3)[0], 430, 325, 32], strides=2)
    # uconv4 = conv2d(uconv4, weights['wrc4'], biases['rcb4'])
    uconv4 = tf.concat([uconv4, conv6], axis=3)
    conv28 = conv2d(uconv4, weights['wc28'], biases['b28'])
    conv29 = conv2d(conv28, weights['wc29'], biases['b29'])
    conv30 = conv2d(conv29, weights['wc30'], biases['b30'])

    # up block 5
    uconv5 = conv2d_transpose(conv30, weights['wuc5'], biases['ub5'], [tf.shape(conv3)[0], 860, 650, 32], strides=2)
    # uconv5 = conv2d(uconv5, weights['wrc5'], biases['rcb5'])
    uconv5 = tf.concat([uconv5, conv3], axis=3)
    conv31 = conv2d(uconv5, weights['wc31'], biases['b31'])
    conv32 = conv2d(conv31, weights['wc32'], biases['b32'])
    conv33 = conv2d(conv32, weights['wc33'], biases['b33'])


    conv34 = conv2d(conv33, weights['wc34'], biases['b34'])
    out = conv34
    return out


def wts_and_bias():
    weights = {'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc2': tf.get_variable('W1', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc3': tf.get_variable('W2', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc4': tf.get_variable('W3', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc5': tf.get_variable('W4', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc6': tf.get_variable('W5', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc7': tf.get_variable('W6', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc8': tf.get_variable('W7', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc9': tf.get_variable('W8', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc10': tf.get_variable('W9', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc11': tf.get_variable('W10', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc12': tf.get_variable('W11', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc13': tf.get_variable('W12', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc14': tf.get_variable('W13', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc15': tf.get_variable('W14', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc16': tf.get_variable('W15', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc17': tf.get_variable('W16', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc18': tf.get_variable('W17', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc1': tf.get_variable('W18', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               # 'wrc1': tf.get_variable('W19', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc19': tf.get_variable('W20', shape=(3, 3, 64, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc20': tf.get_variable('W21', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc21': tf.get_variable('W22', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc2': tf.get_variable('W23', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               # 'wrc2': tf.get_variable('W24', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc22': tf.get_variable('W25', shape=(3, 3, 64, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc23': tf.get_variable('W26', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc24': tf.get_variable('W27', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc3': tf.get_variable('W28', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               # 'wrc3': tf.get_variable('W29', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc25': tf.get_variable('W30', shape=(3, 3, 64, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc26': tf.get_variable('W31', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc27': tf.get_variable('W32', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc4': tf.get_variable('W33', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               # 'wrc4': tf.get_variable('W34', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc28': tf.get_variable('W35', shape=(3, 3, 64, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc29': tf.get_variable('W36', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc30': tf.get_variable('W37', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wuc5': tf.get_variable('W38', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               # 'wrc5': tf.get_variable('W39', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc31': tf.get_variable('W40', shape=(3, 3, 64, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc32': tf.get_variable('W41', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc33': tf.get_variable('W42', shape=(3, 3, 32, 32), initializer=tf.contrib.layers.xavier_initializer()),
               'wc34': tf.get_variable('W43', shape=(1, 1, 32, 1), initializer=tf.contrib.layers.xavier_initializer())}

    biases = {'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b3': tf.get_variable('B2', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b4': tf.get_variable('B3', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b5': tf.get_variable('B4', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b6': tf.get_variable('B5', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b7': tf.get_variable('B6', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b8': tf.get_variable('B7', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b9': tf.get_variable('B8', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b10': tf.get_variable('B9', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b11': tf.get_variable('B10', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b12': tf.get_variable('B11', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b13': tf.get_variable('B12', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b14': tf.get_variable('B13', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b15': tf.get_variable('B14', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b16': tf.get_variable('B15', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b17': tf.get_variable('B16', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b18': tf.get_variable('B17', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'ub1': tf.get_variable('B18', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              # 'rcb1': tf.get_variable('B19', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b19': tf.get_variable('B20', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b20': tf.get_variable('B21', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b21': tf.get_variable('B22', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'ub2': tf.get_variable('B23', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              # 'rcb2': tf.get_variable('B24', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b22': tf.get_variable('B25', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b23': tf.get_variable('B26', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b24': tf.get_variable('B27', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'ub3': tf.get_variable('B28', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              # 'rcb3': tf.get_variable('B29', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b25': tf.get_variable('B30', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b26': tf.get_variable('B31', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b27': tf.get_variable('B32', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'ub4': tf.get_variable('B33', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              # 'rcb4': tf.get_variable('B34', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b28': tf.get_variable('B35', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b29': tf.get_variable('B36', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b30': tf.get_variable('B37', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'ub5': tf.get_variable('B38', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              # 'rcb5': tf.get_variable('B39', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b31': tf.get_variable('B40', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b32': tf.get_variable('B41', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b33': tf.get_variable('B42', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
              'b34': tf.get_variable('B43', shape=(1), initializer=tf.contrib.layers.xavier_initializer()),}

    return weights, biases