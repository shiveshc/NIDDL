import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from cnn_archs import *

def load_data(data_path, all_gt_img_data, all_noisy_data):
    gt_img_path = data_path + '/' + 'gt_imgs'
    noisy_img_path = data_path + '/' + 'noisy_imgs'

    stack_list = os.listdir(gt_img_path)
    stack_num = [int(stack[4:]) for stack in stack_list]
    num_stacks = max(stack_num)

    for stack in range(1, num_stacks + 1):
        curr_img_gt = []
        stack_path_gt = gt_img_path + '/img_' + str(stack)
        curr_img_noisy = []
        stack_path_noisy = noisy_img_path + '/img_' + str(stack)

        zplane_list = os.listdir(stack_path_gt)
        zplane_num = [int(zplane[2:len(zplane) - 4]) for zplane in zplane_list]
        max_zplane_num = max(zplane_num)
        for num in range(max_zplane_num):
            curr_gt_z = cv2.imread(stack_path_gt + '/z_' + str(num + 1) + '.tif', -1)
            curr_img_gt.append(curr_gt_z)

            curr_noisy_z = cv2.imread(stack_path_noisy + '/z_' + str(num + 1) + '.tif', -1)
            curr_img_noisy.append(curr_noisy_z)

        curr_img_gt = np.array(curr_img_gt)
        curr_img_noisy = np.array(curr_img_noisy)
        curr_img_gt = np.transpose(curr_img_gt, axes=(1, 2, 0))
        curr_img_noisy = np.transpose(curr_img_noisy, axes=(1, 2, 0))
        all_gt_img_data.append(curr_img_gt)
        all_noisy_data.append(curr_img_noisy)

    return all_gt_img_data, all_noisy_data


def load_data_maxproj(data_path, all_gt_img_data, all_noisy_data):
    gt_img_path = data_path + '/' + 'gt_imgs'
    noisy_img_path = data_path + '/' + 'noisy_imgs'

    stack_list = os.listdir(gt_img_path)
    stack_num = [int(stack[4:]) for stack in stack_list]
    num_stacks = max(stack_num)

    for stack in range(1, num_stacks + 1):
        curr_img_gt = []
        stack_path_gt = gt_img_path + '/img_' + str(stack)
        curr_img_noisy = []
        stack_path_noisy = noisy_img_path + '/img_' + str(stack)

        zplane_list = os.listdir(stack_path_gt)
        zplane_num = [int(zplane[2:len(zplane) - 4]) for zplane in zplane_list]
        max_zplane_num = max(zplane_num)
        for num in range(max_zplane_num):
            curr_gt_z = cv2.imread(stack_path_gt + '/z_' + str(num + 1) + '.tif', -1)
            curr_img_gt.append(curr_gt_z)

            curr_noisy_z = cv2.imread(stack_path_noisy + '/z_' + str(num + 1) + '.tif', -1)
            curr_img_noisy.append(curr_noisy_z)

        curr_img_gt = np.array(curr_img_gt)
        curr_img_noisy = np.array(curr_img_noisy)
        curr_img_gt = np.transpose(curr_img_gt, axes=(1, 2, 0))
        curr_img_noisy = np.transpose(curr_img_noisy, axes=(1, 2, 0))
        curr_img_gt = np.amax(curr_img_gt, axis=2, keepdims= True)
        curr_img_noisy = np.amax(curr_img_noisy, axis=2, keepdims= True)
        all_gt_img_data.append(curr_img_gt)
        all_noisy_data.append(curr_img_noisy)

    return all_gt_img_data, all_noisy_data


def get_depth_chunks_from_stack(img, depth):
    depth_chunks_img = []
    for z in range(img.shape[2]):
        curr_image = []
        below_frame_num = [n for n in range(int(z - (depth - 1) / 2), int(z))]
        for below_frames in below_frame_num:
            if below_frames < 0:
                curr_image.append(np.zeros((img.shape[0], img.shape[1])))
            else:
                curr_image.append(img[:, :, below_frames])

        curr_image.append(img[:, :, z])

        above_frame_num = [n for n in range(int(z + 1), int(z + (depth - 1) / 2 + 1))]
        for above_frames in above_frame_num:
            if above_frames > img.shape[2] - 1:
                curr_image.append(np.zeros((img.shape[0], img.shape[1])))
            else:
                curr_image.append(img[:, :, above_frames])

        curr_image = np.array(curr_image)
        curr_image = np.transpose(curr_image, axes=(1, 2, 0))
        depth_chunks_img.append(curr_image)

    return depth_chunks_img


def prepare_training_data(all_gt_img_data, all_noisy_img_data, depth):
    train_gt_img_data = []
    train_noisy_img_data = []
    for n in range(len(all_gt_img_data)):
        curr_gt_img = all_gt_img_data[n]
        curr_noisy_img = all_noisy_img_data[n]
        depth_gt_img = get_depth_chunks_from_stack(curr_gt_img, 1)
        depth_noisy_img = get_depth_chunks_from_stack(curr_noisy_img, depth)
        train_gt_img_data.extend(depth_gt_img)
        train_noisy_img_data.extend(depth_noisy_img)

    train_gt_img_data = np.array(train_gt_img_data)
    train_noisy_img_data = np.array(train_noisy_img_data)
    return train_gt_img_data, train_noisy_img_data

def get_patches(img, size, stride):
    x_len = img.shape[1]
    y_len = img.shape[0]

    all_patches = []

    stride_x_cnt = 0
    x_start = 0 + stride_x_cnt * stride
    x_end = x_start + size
    while x_end <= x_len:
        stride_y_cnt = 0
        y_start = 0 + stride_y_cnt * stride
        y_end = y_start + size
        while y_end <= y_len:
            curr_patch = img[y_start:y_end, x_start:x_end]
            all_patches.append(curr_patch)
            stride_y_cnt = stride_y_cnt + 1
            y_start = 0 + stride_y_cnt * stride
            y_end = y_start + size
        stride_x_cnt = stride_x_cnt + 1
        x_start = 0 + stride_x_cnt * stride
        x_end = x_start + size

    return all_patches

def to_patches(train_gt_img_data, train_noisy_img_data):
    train_gt_img_data_patch = []
    train_noisy_img_data_patch = []

    for i in range(train_gt_img_data.shape[0]):
        curr_gt = train_gt_img_data[i, :, :, :]
        curr_gt_patch = get_patches(curr_gt, 128, 64)
        train_gt_img_data_patch.extend(curr_gt_patch)
        curr_noisy = train_noisy_img_data[i, :, :, :]
        curr_noisy_patch = get_patches(curr_noisy, 128, 64)
        train_noisy_img_data_patch.extend(curr_noisy_patch)

    train_gt_img_data_patch = np.array(train_gt_img_data_patch)
    train_noisy_img_data_patch = np.array(train_noisy_img_data_patch)

    return train_gt_img_data_patch, train_noisy_img_data_patch

def NormalizeImage(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def plot_example(gt_img, noisy_img, depth):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
    ax1.imshow(noisy_img[:, :, int((depth+1)/2 - 1)])
    ax2.imshow(gt_img[:, :, 0])
    plt.show()


def split_train_test(X, Y, split_ratio):
    idx = [i for i in range(X.shape[0])]
    random.shuffle(idx)
    test_size = round(X.shape[0]*split_ratio)
    train_X = X[idx[:X.shape[0] - test_size], :, :, :]
    train_Y = Y[idx[:X.shape[0] - test_size], :, :, :]
    test_X = X[idx[X.shape[0] - test_size:], :, :, :]
    test_Y = Y[idx[X.shape[0] - test_size:], :, :, :]

    return train_X, train_Y, test_X, test_Y


def get_cnn_arch_from_argin(name):
    arch_dic = {'unet_fixed': unet_v,
                'unet': unet_v2,
                'unet_v': unet_v,
                'unet_v2': unet_v2,
                'hourglass_wres': hourglass_wres,
                'hourglass_wores': hourglass_wores,
                'unet_v_k5': unet_v_k5,
                'hourglass_wres_k5': hourglass_wres_k5,
                'unet_v_patch': unet_v_patch,
                'hourglass_wres_patch': hourglass_wres_patch}

    return arch_dic[name]