import torch
import time
import shutil
import argparse
import sys
from config import *
from inputs import *
from utils_inference import *
from utils import get_cnn_arch_from_argin
import pickle
import os
from tqdm import tqdm

class ModelConfig(TrainConfig):
    def __init__(self, inputs:dict) -> None:
        super().__init__()
        for param, value in inputs.items():
            setattr(self, param, value)


class InfConfig(InferenceConfig):
    def __init__(self, inputs:dict) -> None:
        super().__init__()
        for param, value in inputs.items():
            setattr(self, param, value)


def get_trained_model_config(
    inference_config        
):
    config_path = os.path.join(inference_config.run, 'model_config.pickle')
    with open(config_path, 'rb') as handle:
        config = pickle.load(handle)
    model_config = ModelConfig(config)
    return model_config


def get_trained_model(
        inference_config,
        model_config
):
    model_path = os.path.join(inference_config.run, 'model_weights.pt')
    model = get_cnn_arch_from_argin(model_config.arch)(model_config.in_channels, model_config.out_channels)
    model.load_state_dict(torch.load(model_path))
    return model


def denoise_data(
        inference_config,
        model_config,
        model,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for d in tqdm(range(len(inference_config.data))):
        data_path = inference_config.data[d]

        print(f'Denoising {data_path}')
        if os.path.isdir(data_path): # data path are directories containing images
            all_noisy_data = load_data(data_path, model_config.max_proj)

            pred_dir = os.path.join(data_path, 'pred_model')
            if os.path.isdir(pred_dir):
                shutil.rmtree(pred_dir)
            os.mkdir(pred_dir)
            file = open(os.path.join(pred_dir, 'inference_runtime.txt'), 'w')
            for i in tqdm(range(len(all_noisy_data))):
                curr_img = all_noisy_data[i]
                curr_img_dir = os.path.join(pred_dir, f'img_{i + 1}')
                os.mkdir(curr_img_dir)
                for z in range(curr_img.shape[2]):
                    input_x = make_cnn_input_data(curr_img, z, int(model_config.depth))
                    input_x = input_x[np.newaxis, :, :, :]
                    input_x = pytorch_specific_manipulations(input_x)
                    input_x = torch.tensor(input_x, dtype=torch.float).to(device)
                    tic = time.time()
                    with torch.no_grad():
                        temp_pred = model(input_x)
                    toc = time.time()
                    temp_pred = temp_pred.cpu().numpy()
                    file.write(f'{i},{z+1},{toc-tic},{model_config.arch},{model_config.depth},{model_config.run},{model_config.tsize}\n')
                    cv2.imwrite(os.path.join(curr_img_dir, f'z_{z + 1}.tif'), temp_pred[0, int((model_config.depth + 1) / 2 - 1), :, :].astype(np.uint16))
            file.close()

        elif os.path.exists(data_path): # data path are image files
            all_noisy_data, all_img_name, all_img_path = load_data_indiv_imgs(data_path, model_config.max_proj)

            pred_dir = os.path.join(all_img_path[0], f'pred_{all_img_name[0]}')
            if os.path.isdir(pred_dir):
                shutil.rmtree(pred_dir)
            os.mkdir(pred_dir)
            file = open(os.path.join(pred_dir, 'inference_runtime.txt'), 'w')
            curr_img = all_noisy_data[0]
            if len(curr_img.shape) == 3:
                for z in range(curr_img.shape[2]):
                    input_x = make_cnn_input_data(curr_img, z, int(model_config.depth))
                    input_x = input_x[np.newaxis, :, :, :]
                    input_x = pytorch_specific_manipulations(input_x)
                    input_x = torch.tensor(input_x, dtype=torch.float).to(device)
                    tic = time.time()
                    with torch.no_grad():
                        temp_pred = model(input_x)
                    toc = time.time()
                    temp_pred = temp_pred.cpu().numpy()
                    file.write(f'{z + 1},{toc-tic},{model_config.arch},{model_config.depth},{model_config.run},{model_config.tsize}\n')
                    cv2.imwrite(os.path.join(pred_dir, f'z_{z + 1}.tif'), temp_pred[0, int((model_config.depth + 1) / 2 - 1), :, :].astype(np.uint16))
            elif len(curr_img.shape) == 2:
                input_x = curr_img[:, :, np.newaxis]
                input_x = make_cnn_input_data(input_x, 0, int(model_config.depth))
                input_x = input_x[np.newaxis, :, :, :]
                input_x = pytorch_specific_manipulations(input_x)
                input_x = torch.tensor(input_x, dtype=torch.float).to(device)
                tic = time.time()
                with torch.no_grad():
                        temp_pred = model(input_x)
                toc = time.time()
                temp_pred = temp_pred.cpu().numpy()
                file.write(f'{1},{toc-tic},{model_config.arch},{model_config.depth},{model_config.run},{model_config.tsize}\n')
                cv2.imwrite(os.path.join(pred_dir, f'{all_img_name[0]}.tif'), temp_pred[0, int((model_config.depth + 1) / 2 - 1), :, :].astype(np.uint16))
            file.close()



if __name__ == '__main__':

    # get inference parameters
    inference_config = InfConfig(inference_arg_parser())

    # get saved model
    model_config = get_trained_model_config(inference_config)
    model = get_trained_model(inference_config, model_config)
    
    # denoise data
    denoise_data(inference_config, model_config, model)
    