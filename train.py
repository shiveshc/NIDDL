import tensorflow as tf
import sys
import argparse
import time
import shutil
from utils import *

def parse_argument(arg_list):
    if not arg_list:
        arg_list = ['-h']
        print('error - input required, see description below')

    parser = argparse.ArgumentParser(prog='train.py', description= 'train CNN model to denoise volumetric functional recordings')
    parser.add_argument('data', help= 'training data path')
    parser.add_argument('run', type=int, help='run number to distinguish different runs')
    parser.add_argument('max_proj', type=int, choices= [1, 0], help= '1 if train network on max projection of 3D stacks else 0', default= 0)
    parser.add_argument('-out', help= 'location for saving results')
    parser.add_argument('-arch', choices=['unet',
                                         'unet_fixed',
                                         'hourglass_wres',
                                         'hourglass_wores'], help= 'CNN architechture to use for training (default is hourglass_wres)', default= 'hourglass_wres')
    parser.add_argument('-mode', choices= ['2D', '2.5D', '3D'], help= 'training mode (default is 2D)', default= '2D')
    parser.add_argument('-depth', type= int, help= 'stack depth to use for training (must be odd number, default is 1)', default= 1)
    parser.add_argument('-loss', choices= ['l2', 'l1'], help= 'L2 or L1 loss for training (default is l1)', default= 'l1')
    parser.add_argument('-epochs', type= int, help= 'number of epochs to train the model for (150-200 is good choice, default is 150)', default= 150)
    parser.add_argument('-lr', type= float, help= 'learning rate (default is 0.001)', default= 0.001)
    parser.add_argument('-bs', type= int, help= 'batch size of training (default is 6)', default= 6)
    parser.add_argument('-tsize', type= int, help= 'data size (number of images) to use for training')
    args = parser.parse_args(arg_list)
    return args.data,\
           args.run,\
           args.max_proj, \
           args.out, \
           args.arch, \
           args.mode, \
           args.depth,\
           args.loss, \
           args.epochs,\
           args.lr, \
           args.bs, \
           args.tsize


if __name__ == '__main__':
    # parse input
    data_path, run, max_proj, out_path, arch_name, mode, depth, loss, epochs, lr, bs, tsize = parse_argument(sys.argv[1:])
    if max_proj == 1:
        assert mode == '2D', 'training model on max-projection images thus training mode should be 2D'

    if mode == '2D':
        assert depth == 1, 'for 2D training mode, stack depth for training must be 1'
    else:
        assert depth%2 == 1, 'for 2.5D or 3D training mode, stack depth must be an odd number'

    assert os.path.isdir(data_path), 'training data path does not exist'


    ## load data
    # base_data_path = ['D:/Shivesh/Denoising/20210206_denoising_ZC392/cond_10ms110_10ms1000_v2']
    base_data_path = [data_path]
    all_gt_img_data = []
    all_noisy_data = []
    for paths in base_data_path:
        all_gt_img_data, all_noisy_data = load_data(paths, all_gt_img_data, all_noisy_data, max_proj)


    ## prepare training data
    train_gt_img_data, train_noisy_data = prepare_training_data(all_gt_img_data, all_noisy_data, depth, mode)

    ## split data into patches
    # train_gt_img_data_patch, train_noisy_img_data_patch = to_patches(train_gt_img_data, train_noisy_data)
    train_gt_img_data_patch = train_gt_img_data
    train_noisy_img_data_patch = train_noisy_data

    ## subsample data based on tsize for training
    if tsize != None:
        # if tsize is specified, use tsize number of images as train and rest as test
        tsize = int(tsize)
        split_ratio = (train_gt_img_data_patch.shape[0] - tsize)/train_gt_img_data_patch.shape[0]
        train_X, train_Y, test_X, test_Y = split_train_test(train_noisy_img_data_patch, train_gt_img_data_patch, split_ratio)
        tsize = train_X.shape[0]
    else:
        # if tsize is not specified, split full data for train and test
        train_X, train_Y, test_X, test_Y = split_train_test(train_noisy_img_data_patch, train_gt_img_data_patch, 0.33)
        tsize = train_X.shape[0]


    ## define CNN model
    training_iters = epochs
    learning_rate = lr
    batch_size = bs

    x = tf.placeholder("float", [None, train_noisy_img_data_patch.shape[1], train_noisy_img_data_patch.shape[2], train_noisy_img_data_patch.shape[3]])
    y = tf.placeholder("float", [None, train_gt_img_data_patch.shape[1], train_gt_img_data_patch.shape[2], train_gt_img_data_patch.shape[3]])
    arch = get_cnn_arch_from_argin(arch_name)
    output_shape = y.shape[3]
    pred = arch.conv_net(x, output_shape)

    if loss == 'l1':
        cost = tf.reduce_mean(tf.abs(pred - y))
    elif loss == 'l2':
        cost = tf.reduce_mean(tf.squared_difference(pred, y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    ## make folder where all results will be saved
    if out_path != None:
        if os.path.isdir(out_path) == False:
            os.mkdir(out_path)
        results_dir = out_path + '/run_' + arch_name + '_' + loss + '_mp' + str(max_proj) + '_m' + mode + '_d' + str(depth) + '_' + str(run) + '_' + str(tsize)
    else:
        results_dir = 'run_' + arch_name + '_' + loss + '_mp' + str(max_proj) + '_m' + mode + '_d' + str(depth) + '_' + str(run) + '_' + str(tsize)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    ## train model and save results
    with tf.Session() as sess:
        sess.run(init)
        train_loss = []
        test_loss = []
        summary_writer = tf.summary.FileWriter(results_dir, sess.graph)
        file = open(results_dir + '/training_loss.txt', 'a')

        for i in range(training_iters):
            if tsize > 500:
                idx = random.sample(range(tsize), 500)
                curr_batch_X = train_X[idx, :, :, :]
                curr_batch_Y = train_Y[idx, :, :, :]
            else:
                curr_batch_X = train_X
                curr_batch_Y = train_Y
            tic = time.clock()
            for batch in range(len(curr_batch_X) // batch_size):
                batch_x = curr_batch_X[batch * batch_size:min((batch + 1) * batch_size, len(curr_batch_X))]
                batch_y = curr_batch_Y[batch * batch_size:min((batch + 1) * batch_size, len(curr_batch_Y))]
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            toc = time.clock()

            # Calculate accuracy of 10 test images, repeat 5 times and report mean
            batch_test_loss = []
            for k in range(5):
                idx = [i for i in range(test_X.shape[0])]
                random.shuffle(idx)
                curr_loss = sess.run(cost, feed_dict={x: test_X[idx[:min(10, test_X.shape[0])], :, :, :], y: test_Y[idx[:min(10, test_X.shape[0])], :, :, :]})
                batch_test_loss.append(curr_loss)

            # Calculate accuracy of 10 train images, repeat 5 times and report mean
            batch_train_loss = []
            for k in range(5):
                idx = [i for i in range(train_X.shape[0])]
                random.shuffle(idx)
                curr_loss = sess.run(cost, feed_dict={x: train_X[idx[:min(10, train_X.shape[0])], :, :, :], y: train_Y[idx[:min(10, train_X.shape[0])], :, :, :]})
                batch_train_loss.append(curr_loss)

            mean_train_loss = sum(batch_train_loss) / len(batch_train_loss)
            mean_test_loss = sum(batch_test_loss) / len(batch_test_loss)
            print("Iter " + str(i) + ", Train Loss= " + "{:.6f}".format(mean_train_loss) + ",Test Loss= " + "{:.6f}".format(mean_test_loss))
            file.write(str(i) + ',' + str(mean_train_loss) + ',' + str(mean_test_loss) + ',' + str(toc-tic) + ',' + str(depth) + ',' + str(run) + ',' + str(tsize) + '\n')

        file.close()

        # save final model
        saver.save(sess, results_dir + '/model')

        # save some random prediction examples
        for i in range(10):
            temp_idx = random.randint(0, test_X.shape[0])
            temp_X = test_X[temp_idx, :, :, :]
            temp_Y = test_Y[temp_idx, :, :, :]
            temp_X = temp_X[np.newaxis, :, :, :]
            temp_Y = temp_Y[np.newaxis, :, :, :]
            temp_pred = sess.run(pred, feed_dict={x: temp_X, y: temp_Y})
            if mode == '2D':
                cv2.imwrite(results_dir + '/X_' + str(temp_idx + 1) + '.png', temp_X[0, :, :, 0].astype(np.uint16))  # this is the middle zplane corresponding to gt zplane
                cv2.imwrite(results_dir + '/Y_' + str(temp_idx + 1) + '.png', temp_Y[0, :, :, 0].astype(np.uint16))
                cv2.imwrite(results_dir + '/pred_' + str(temp_idx + 1) + '.png', temp_pred[0, :, :, 0].astype(np.uint16))
            elif mode == '2.5D':
                cv2.imwrite(results_dir + '/X_' + str(temp_idx + 1) + '.png', temp_X[0, :, :, int((depth + 1) / 2 - 1)].astype(np.uint16))  # this is the middle zplane corresponding to gt zplane
                cv2.imwrite(results_dir + '/Y_' + str(temp_idx + 1) + '.png', temp_Y[0, :, :, 0].astype(np.uint16))
                cv2.imwrite(results_dir + '/pred_' + str(temp_idx + 1) + '.png', temp_pred[0, :, :, 0].astype(np.uint16))
            elif mode == '3D':
                os.mkdir(results_dir + '/img_' + str(temp_idx + 1))
                for z in range(temp_pred.shape[3]):
                    cv2.imwrite(results_dir + '/img_' + str(temp_idx + 1) + '/X_z' + str(z + 1) + '.png', temp_X[0, :, :, z].astype(np.uint16))  # this is the middle zplane corresponding to gt zplane
                    cv2.imwrite(results_dir + '/img_' + str(temp_idx + 1) + '/Y_z' + str(z + 1) + '.png', temp_Y[0, :, :, z].astype(np.uint16))
                    cv2.imwrite(results_dir + '/img_' + str(temp_idx + 1) + '/pred_z' + str(z + 1) + '.png', temp_pred[0, :, :, z].astype(np.uint16))

        # calculate accuracy on test data
        file = open(results_dir + '/test_data_loss.txt', 'a')
        idx = [i for i in range(test_X.shape[0])]
        random.shuffle(idx)
        for i in range(min(150, len(idx))):
            temp_X = test_X[idx[i], :, :, :]
            temp_Y = test_Y[idx[i], :, :, :]
            temp_X = temp_X[np.newaxis, :, :, :]
            temp_Y = temp_Y[np.newaxis, :, :, :]
            curr_loss = sess.run(cost, feed_dict={x: temp_X, y: temp_Y})
            tic = time.clock()
            temp_pred = sess.run(pred, feed_dict={x: temp_X, y: temp_Y})
            toc = time.clock()
            file.write(str(i) + ',' + str(idx[i] + 1) + ',' + str(curr_loss) + ',' + str(toc-tic) + ',' + str(depth) + ',' + str(run) + ',' + str(tsize) + '\n')

        file.close()

        summary_writer.close()
