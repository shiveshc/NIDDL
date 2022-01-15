import os
import cv2
import numpy as np

def load_data(data_path):
    noisy_vid_path = data_path + '/' + 'noisy_imgs'

    all_noisy_data = []

    img_list = os.listdir(noisy_vid_path)
    img_num = [int(img[4:len(img)]) for img in img_list]
    num_imgs = max(img_num)

    for img in range(1, num_imgs + 1):
        zplane_list = os.listdir(noisy_vid_path + '/img_' + str(img))
        zplane_num = [int(zplane[2:len(zplane) - 4]) for zplane in zplane_list]
        num_zplanes = max(zplane_num)
        curr_noisy_img = []
        for z in range(1, num_zplanes + 1):
            noisy_img = cv2.imread(noisy_vid_path + '/img_' + str(img) + '/z_' + str(z) + '.tif', -1)
            curr_noisy_img.append(noisy_img)

        curr_noisy_img = np.array(curr_noisy_img)
        curr_noisy_img = np.transpose(curr_noisy_img, axes=(1, 2, 0))
        all_noisy_data.append(curr_noisy_img)

    all_noisy_data = np.array(all_noisy_data)

    return all_noisy_data

def load_data_maxproj(data_path):
    noisy_vid_path = data_path + '/' + 'noisy_imgs'

    all_noisy_data = []

    img_list = os.listdir(noisy_vid_path)
    img_num = [int(img[4:len(img)]) for img in img_list]
    num_imgs = max(img_num)

    for tp in range(1, num_imgs + 1):
        curr_tp_noisy = []
        tp_path = noisy_vid_path + '/img_' + str(tp)
        zplane_list = os.listdir(tp_path)
        zplane_num = [int(zplane[2:len(zplane) - 4]) for zplane in zplane_list]
        num_zplanes = max(zplane_num)
        for z in range(1, num_zplanes + 1):
            curr_noisy_img = cv2.imread(tp_path + '/z_' + str(z) + '.tif', -1)
            curr_tp_noisy.append(curr_noisy_img)

        curr_tp_noisy = np.array(curr_tp_noisy)
        curr_tp_noisy = np.transpose(curr_tp_noisy, axes= (1, 2, 0))
        curr_tp_noisy = np.amax(curr_tp_noisy, axis= 2, keepdims= True)
        all_noisy_data.append(curr_tp_noisy)

    all_noisy_data = np.array(all_noisy_data)
    return all_noisy_data

def make_cnn_input_data(curr_tp, znum, depth):
    input_to_cnn = []
    below_frame_num = [n for n in range(int(znum - (depth - 1) / 2), int(znum))]
    for below_frames in below_frame_num:
        if below_frames < 0:
            input_to_cnn.append(np.zeros((curr_tp.shape[0], curr_tp.shape[1])))
        else:
            input_to_cnn.append(curr_tp[:, :, below_frames])

    input_to_cnn.append(curr_tp[:, :, znum])

    above_frame_num = [n for n in range(int(znum + 1), int(znum + (depth - 1) / 2 + 1))]
    for above_frames in above_frame_num:
        if above_frames > curr_tp.shape[2] - 1:
            input_to_cnn.append(np.zeros((curr_tp.shape[0], curr_tp.shape[1])))
        else:
            input_to_cnn.append(curr_tp[:, :, above_frames])

    input_to_cnn = np.array(input_to_cnn)
    input_to_cnn = np.transpose(input_to_cnn, axes=(1, 2, 0))
    return input_to_cnn