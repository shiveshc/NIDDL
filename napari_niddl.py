import tifffile
from magicgui import magic_factory, magicgui
from typing import Annotated, Literal
import tifffile
import napari
import pathlib
import os
import cv2

from inference import InfConfig, get_trained_model_config, get_trained_model, denoise_data



def get_denoised_img_path(noisy_img_path):
    path_split = str(noisy_img_path).split(os.sep)
    denoised_img_path = os.path.join(os.path.sep, *path_split[0:len(path_split)-1], f'pred_{path_split[-1]}', f'{path_split[-1]}.tif')

    return denoised_img_path


def denoise(noisy_img_path, datatype):

    if datatype == 'Synthetic':
        run_path = 'test_runs/run_hourglass_wres_l1_mp1_m2D_d1_1_0'

    inference_config = {'data': [noisy_img_path], 'run': run_path}

    inference_config = InfConfig(inference_config)
    trained_model_config = get_trained_model_config(inference_config)
    trained_model = get_trained_model(inference_config, trained_model_config)
    denoise_data(inference_config, trained_model_config, trained_model)

    denoised_img_path = get_denoised_img_path(noisy_img_path)
    denoised_img = tifffile.imread(denoised_img_path)

    return denoised_img


@magicgui(
    call_button="Denoise",
    datatype={"name": "DataType", "choices": ['WholeBrain', 'VentralCord', 'Neurite', 'Synthetic']},
)
def widget(
    datatype='WholeBrain',
    noisy_img_path=pathlib.Path('/noisy_img_path/img.tif'),
    gt_img_path=pathlib.Path('/gt_img_path/img.tif'), 
):
    viewer = napari.current_viewer()
    
    if os.path.isfile(noisy_img_path):
        noisy_img = tifffile.imread(noisy_img_path)
        viewer.add_image(noisy_img, name='noisy_img', visible=False)
    else:
        raise FileNotFoundError(f"{noisy_img_path} is not found")
    
    if os.path.isfile(gt_img_path):
        gt_img = tifffile.imread(gt_img_path)
        viewer.add_image(gt_img, name='gt_img', visible=False)
    
    denoised_img = denoise(noisy_img_path, datatype)
    viewer.add_image(denoised_img, name='denoised_img')


viewer = napari.Viewer()
# img = cv2.imread('extra/denoise_traces.png')
# viewer.add_image(img, name='niddl')
# denoise_widget = magicgui(widget)
viewer.window.add_dock_widget(widget, name= 'NIDDL Denoise', area='right')
napari.run()
    
