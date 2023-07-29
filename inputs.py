import argparse
import sys
from config import TrainConfig, InferenceConfig
import os

def train_arg_parser():

    default_config = TrainConfig()
    default_param = vars(default_config)

    parser = argparse.ArgumentParser(description='train CNN model to denoise volumetric functional recordings')
    for att in default_param:
        if type(default_param[att]) == list:
            parser.add_argument(f'-{att}', nargs='+', default=default_param[att])
        else:
            parser.add_argument(f'-{att}', type=type(default_param[att]), default=default_param[att])
    
    args = parser.parse_args(sys.argv[1:])

    assert len(args.data) != 0, 'please provide full training data path e.g. train_pytorch.py -data "/home/user/data"'
    assert all([os.path.isdir(p) for p in args.data]), 'training data path does not exist'
    assert args.max_proj in [0, 1], f'max_proj must be 0 if train network on 3D images or 1 if train on max projection of 3D images'
    assert args.arch in ['unet', 'unet_fixed', 'hourglass_wores', 'hourglass_wres'], f'choices for arch are - unet, unet_fixed, hourglass_wores, hourglass_wres'
    assert args.mode in ['2D', '2.5D', '3D'], f'choices for mode are - 2D, 2.5D, 3D'
    assert args.depth%2 == 1, f'depth must be an odd number'
    assert args.loss in ['l2', 'l1'], f'choices for loss are - l2, l1'

    if args.max_proj == 1:
        assert args.mode == '2D', f'training model on max-projection images thus training mode should be 2D'
    
    if args.mode == '2D':
        assert args.depth == 1, 'for 2D training mode, stack depth for training must be 1'

    return vars(args)


def inference_arg_parser():
    default_config = InferenceConfig()
    default_param = vars(default_config)

    parser = argparse.ArgumentParser(description='inference CNN model to denoise volumetric functional recordings')
    for att in default_param:
        if type(default_param[att]) == list:
            parser.add_argument(f'-{att}', nargs='+', default=default_param[att])
        else:
            parser.add_argument(f'-{att}', type=type(default_param[att]), default=default_param[att])
    
    args = parser.parse_args(sys.argv[1:])

    assert args.data != [], 'please provide full data path to be denoised e.g. inference_pytorch.py -data /home/user/data -run /home/user/model/'
    assert args.run != '', 'please provide path of trained model weights e.g. inference_pytorch.py -data /home/user/data -run /home/user/model/'
    assert os.path.exists(f'{args.run}/model_weights.pt'), 'trained model either does not exist in run path or not correctly saved'
    assert os.path.exists(f'{args.run}/model_config.pickle'), 'trained model config file either does not exist in run path or not correctly saved'
    return vars(args)


