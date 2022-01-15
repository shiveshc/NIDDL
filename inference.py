import tensorflow as tf
import time
import shutil
from utils_inference import *
from utils import get_cnn_arch_from_argin

def get_run_params(run_path):
    slash_idx = [i for i, v in enumerate(run_path) if v == '/']
    base_path_run = run_path[0:slash_idx[-1]]
    run_name = run_path[slash_idx[-1] + 1::]
    if 'm2D' in run_name:
        model_type = '2D'
    elif 'm2.5D' in run_name:
        model_type = '2.5D'
    elif 'm3D' in run_name:
        model_type = '3D'

    if 'hourglass_wores_' in run_name:
        arch_name = 'hourglass_wores'
    elif 'hourglass_wres_' in run_name:
        arch_name = 'hourglass_wres'
    elif 'unet_v_' in run_name:
        arch_name = 'unet_v'
    elif 'unet_v2_' in run_name:
        arch_name = 'unet_v2'
    elif 'unet_' in run_name:
        arch_name = 'unet'
    elif 'unet_fixed_' in run_name:
        arch_name = 'unet_fixed'

    underscore_idx = [i for i, char in enumerate(run_name) if char == '_']
    depth = run_name[underscore_idx[-3] + 2:underscore_idx[-2]]
    run = run_name[underscore_idx[-2] + 1:underscore_idx[-1]]
    tsize = run_name[underscore_idx[-1] + 1:]

    return base_path_run, run_name, model_type, arch_name, depth, run, tsize


if __name__ == '__main__':

    # model dir name
    run_path = 'D:/Shivesh/Denoising/denoise_cnn/Results/runs_exp_both_16bit_depth/3D/d5/sample_run_hourglass_wres_l1_m2.5D_d1_1_942'
    data_paths = ['D:/Shivesh/Denoising/20210421_magnify_ZIM504/cond_10ms110_10ms1000_v2/sampled_data_d5']

    base_path_run, run_name, model_type, arch_name, depth, run, tsize = get_run_params(run_path)

    save_model_path = run_path

    for data_path in data_paths:
        all_noisy_data = load_data(data_path)

        # redefine cnn model based on parameters extracted from run_name
        tf.reset_default_graph()
        x = tf.placeholder("float", [None, all_noisy_data.shape[1], all_noisy_data.shape[2], depth])
        arch = get_cnn_arch_from_argin(arch_name)

        if model_type == '3D':
            output_shape = int(depth)
        else:
            output_shape = 1

        pred = arch.conv_net(x, output_shape)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            # saved_model.restore(sess, save_model_path + '/model')
            saver.restore(sess, save_model_path + '/model')
            model_size = os.path.getsize(save_model_path + '/model.data-00000-of-00001')

            pred_dir = data_path + '/pred_' + run_name
            if os.path.isdir(pred_dir):
                shutil.rmtree(pred_dir)
            os.mkdir(pred_dir)

            file = open(data_path + '/' + run_name + '_inference_runtime.txt', 'w')
            for i in range(all_noisy_data.shape[0]):
                curr_img_dir = pred_dir + '/img_' + str(i + 1)
                os.mkdir(curr_img_dir)

                input_x = all_noisy_data[i, :, :, :]
                input_x = input_x[np.newaxis, :, :, :]
                tic = time.clock()
                temp_pred = sess.run(pred, feed_dict= {x: input_x})
                toc = time.clock()
                file.write(str(i) + ',' + ',' + str(toc-tic) + ',' + str(model_size) + ',' + arch_name + ',' + str(depth) + ',' + str(run) + ',' + str(tsize) + '\n')
                if model_type == '2D':
                    cv2.imwrite(pred_dir + '/img_' + str(i + 1) + '/z_1.tif', temp_pred[0, :, :, 0].astype(np.uint16))
                elif model_type == '2.5D':
                    cv2.imwrite(pred_dir + '/img_' + str(i + 1) + '/z_' + str(int((int(depth) + 1)/2)) + '.tif', temp_pred[0, :, :, 0].astype(np.uint16))
                elif model_type == '3D':
                    for z in range(temp_pred.shape[3]):
                        cv2.imwrite(pred_dir + '/img_' + str(i + 1) + '/z_' + str(z + 1) + '.tif', temp_pred[0, :, :, z].astype(np.uint16)) # save full 3D prediction


            file.close()