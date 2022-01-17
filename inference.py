import tensorflow as tf
import time
import shutil
import argparse
import sys
from utils_inference import *
from utils import get_cnn_arch_from_argin

def parse_argument(arg_list):
    if not arg_list:
        arg_list = ['-h']
        print('error - input required, see description below')

    parser = argparse.ArgumentParser(prog='inference.py', description= 'denoise images using weights of trained model')
    parser.add_argument('data', nargs= '+', help= 'paths of datasets to be denoised')
    parser.add_argument('run', help='path of saved trained model')
    args = parser.parse_args(arg_list)
    return args.data,\
           args.run


if __name__ == '__main__':

    # parse input arguments
    data_paths, run_path = parse_argument(sys.argv[1:])
    assert os.path.exists(run_path + '/checkpoint'), 'trained model either does not exist in run path or not correctly saved'
    assert os.path.exists(run_path + '/model.index'), 'trained model either does not exist in run path or not correctly saved'
    assert os.path.exists(run_path + '/model.meta'), 'trained model either does not exist in run path or not correctly saved'
    assert os.path.exists(run_path + '/model.data-00000-of-00001'), 'trained model either does not exist in run path or not correctly saved'

    # get run parameters
    base_path_run, run_name, max_proj, mode, arch_name, depth, run, tsize = get_run_params(run_path)

    save_model_path = run_path

    for data_path in data_paths:
        if os.path.isdir(data_path):
            all_noisy_data = load_data(data_path, max_proj)
        elif os.path.exists(data_path):
            all_noisy_data, all_img_name, all_img_path = load_data_indiv_imgs(data_path, max_proj)

        # redefine cnn model based on parameters extracted from run_name
        tf.reset_default_graph()
        x = tf.placeholder("float", [None, all_noisy_data[0].shape[0], all_noisy_data[0].shape[1], depth])
        arch = get_cnn_arch_from_argin(arch_name)

        if mode == '3D':
            output_shape = int(depth)
        else:
            output_shape = 1

        pred = arch.conv_net(x, output_shape)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # saved_model.restore(sess, save_model_path + '/model')
            saver.restore(sess, save_model_path + '/model')
            model_size = os.path.getsize(save_model_path + '/model.data-00000-of-00001')


            if os.path.isdir(data_path):
                pred_dir = data_path + '/pred_' + run_name
                if os.path.isdir(pred_dir):
                    shutil.rmtree(pred_dir)
                os.mkdir(pred_dir)
                file = open(pred_dir + '/' + run_name + '_inference_runtime.txt', 'w')

                for i in range(len(all_noisy_data)):
                    curr_img = all_noisy_data[i]
                    curr_img_dir = pred_dir + '/img_' + str(i + 1)
                    os.mkdir(curr_img_dir)
                    for z in range(curr_img.shape[2]):
                        input_x = make_cnn_input_data(curr_img, z, int(depth))
                        input_x = input_x[np.newaxis, :, :, :]
                        tic = time.clock()
                        temp_pred = sess.run(pred, feed_dict={x: input_x})
                        toc = time.clock()
                        file.write(str(i) + ',' + str(z + 1) + ',' + str(toc - tic) + ',' + str(model_size) + ',' + arch_name + ',' + str(depth) + ',' + str(run) + ',' + str(tsize) + '\n')
                        cv2.imwrite(curr_img_dir + '/z_' + str(z + 1) + '.tif', temp_pred[0, :, :, int((depth + 1) / 2 - 1)].astype(np.uint16))

                file.close()

            elif os.path.exists(data_path):
                pred_dir = all_img_path[0] + '/pred_' + run_name + '_' + all_img_name[0]
                if os.path.isdir(pred_dir):
                    shutil.rmtree(pred_dir)
                os.mkdir(pred_dir)
                file = open(pred_dir + '/' + run_name + '_inference_runtime.txt', 'w')

                curr_img = all_noisy_data[0]
                if len(curr_img.shape) == 3:
                    for z in range(curr_img.shape[2]):
                        input_x = make_cnn_input_data(curr_img, z, int(depth))
                        input_x = input_x[np.newaxis, :, :, :]
                        tic = time.clock()
                        temp_pred = sess.run(pred, feed_dict={x: input_x})
                        toc = time.clock()
                        file.write(str(z + 1) + ',' + str(toc - tic) + ',' + str(model_size) + ',' + arch_name + ',' + str(depth) + ',' + str(run) + ',' + str(tsize) + '\n')
                        cv2.imwrite(pred_dir + '/z_' + str(z + 1) + '.tif', temp_pred[0, :, :, int((depth + 1) / 2 - 1)].astype(np.uint16))
                elif len(curr_img.shape) == 2:
                    input_x = curr_img[:, :, np.newaxis]
                    input_x = make_cnn_input_data(input_x, 0, int(depth))
                    input_x = input_x[np.newaxis, :, :, :]
                    tic = time.clock()
                    temp_pred = sess.run(pred, feed_dict={x: input_x})
                    toc = time.clock()
                    file.write(str(1) + ',' + ',' + str(toc - tic) + ',' + str(model_size) + ',' + arch_name + ',' + str(depth) + ',' + str(run) + ',' + str(tsize) + '\n')
                    cv2.imwrite(pred_dir + '/' + all_img_name[0] + '.tif', temp_pred[0, :, :, int((depth + 1) / 2 - 1)].astype(np.uint16))

                file.close()