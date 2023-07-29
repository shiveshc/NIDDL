import os
import cv2
import numpy as np

def load_data(data_path, max_proj):
    noisy_vid_path = os.path.join(data_path, 'noisy_imgs')

    all_noisy_data = []

    img_list = os.listdir(noisy_vid_path)

    for img in img_list:
        if os.path.isfile(os.path.join(noisy_vid_path, img)):
            curr_noisy_img = cv2.imread(os.path.join(noisy_vid_path, img), -1)
            if len(curr_noisy_img.shape) == 2:
                curr_noisy_img = np.expand_dims(curr_noisy_img, 2)
        elif os.path.isdir(os.path.join(noisy_vid_path, img)):
            zplane_list = os.listdir(os.path.join(noisy_vid_path, img))
            zplane_num = [int(zplane[2:len(zplane) - 4]) for zplane in zplane_list]
            num_zplanes = max(zplane_num)
            curr_noisy_img = []
            for z in range(1, num_zplanes + 1):
                noisy_img = cv2.imread(os.path.join(noisy_vid_path, img, f'z_{z}.tif'), -1)
                curr_noisy_img.append(noisy_img)

            curr_noisy_img = np.array(curr_noisy_img)
            curr_noisy_img = np.transpose(curr_noisy_img, axes=(1, 2, 0))
            if max_proj == 1:
                curr_noisy_img = np.amax(curr_noisy_img, axis=2, keepdims=True)
        all_noisy_data.append(curr_noisy_img)

    return all_noisy_data

# def load_data(data_path, max_proj):
#     noisy_vid_path = os.path.join(data_path, 'noisy_imgs')

#     all_noisy_data = []

#     img_list = os.listdir(noisy_vid_path)
#     img_num = [int(img[4:len(img)]) for img in img_list]
#     num_imgs = max(img_num)

#     for img in range(1, num_imgs + 1):
#         zplane_list = os.listdir(os.path.join(noisy_vid_path, f'img_{img}'))
#         zplane_num = [int(zplane[2:len(zplane) - 4]) for zplane in zplane_list]
#         num_zplanes = max(zplane_num)
#         curr_noisy_img = []
#         for z in range(1, num_zplanes + 1):
#             noisy_img = cv2.imread(os.path.join(noisy_vid_path, f'img_{img}', f'z_{z}.tif'), -1)
#             curr_noisy_img.append(noisy_img)

#         curr_noisy_img = np.array(curr_noisy_img)
#         curr_noisy_img = np.transpose(curr_noisy_img, axes=(1, 2, 0))
#         if max_proj == 1:
#             curr_noisy_img = np.amax(curr_noisy_img, axis=2, keepdims=True)
#         all_noisy_data.append(curr_noisy_img)

#     return all_noisy_data

def load_data_indiv_imgs(img, max_proj):
    all_noisy_data = []
    all_img_name = []
    all_img_path = []

    curr_img_path, curr_img_name = os.path.split(img)
    curr_img = cv2.imread(img, -1)
    if max_proj == 1:
        if len(curr_img.shape) == 3:
            curr_img = np.amax(curr_img, axis= 2, keepdims= True)
    all_noisy_data.append(curr_img)
    all_img_name.append(curr_img_name)
    all_img_path.append(curr_img_path)

    return all_noisy_data, all_img_name, all_img_path

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

def get_run_params(run_path):
    slash_idx = [i for i, v in enumerate(run_path) if v == '/']
    base_path_run = run_path[0:slash_idx[-1]]
    run_name = run_path[slash_idx[-1] + 1::]

    if 'mp0' in run_name:
        max_proj = 0
    elif 'mp1' in run_name:
        max_proj = 1

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
    depth = int(run_name[underscore_idx[-3] + 2:underscore_idx[-2]])
    run = int(run_name[underscore_idx[-2] + 1:underscore_idx[-1]])
    tsize = int(run_name[underscore_idx[-1] + 1:])

    return base_path_run, run_name, max_proj, model_type, arch_name, depth, run, tsize

def pytorch_specific_manipulations(img):
    img = np.transpose(img, (0, 3, 1, 2)).astype('int16')
    return img