import tifffile
from magicgui import magic_factory, magicgui
import tifffile
import napari
import os
from napari.utils.notifications import show_info

from inference import InfConfig, get_trained_model_config, get_trained_model, denoise_data



def get_denoised_img_path(noisy_img_path):
    path_split = noisy_img_path.split(os.sep)
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
    noisy_img_path = dict(widget_type='FileEdit', label='noisy_img: ', tooltip='select noisy image'),
    gt_img_path = dict(widget_type='FileEdit', label='gt_img: ', tooltip='select gt image'),
    load_image  = dict(widget_type='PushButton', text='Load Images', tooltip='load noisy and gt images'),
)
def NIDDL_widget(
    datatype,
    noisy_img_path,
    gt_img_path,
    load_image
):
    viewer = napari.current_viewer()
    noisy_img_path = str(noisy_img_path)
    if noisy_img_path == '' or noisy_img_path == '.':
        show_info('Please Load Images')
    else:
        denoised_img = denoise(noisy_img_path, datatype)
        viewer.add_image(denoised_img, name='denoised_img')


@NIDDL_widget.load_image.changed.connect
def load_data(e):
    viewer = napari.current_viewer()
    
    noisy_img_path = str(NIDDL_widget.noisy_img_path.value)
    gt_img_path = str(NIDDL_widget.gt_img_path.value)

    print(noisy_img_path)
    print(gt_img_path)

    if noisy_img_path == '.':
        show_info('Please select noisy image')
    else:
        if os.path.isfile(noisy_img_path):
            noisy_img = tifffile.imread(noisy_img_path)
            viewer.add_image(noisy_img, name='noisy_img', visible=False)
        else:
            raise FileNotFoundError(f"{noisy_img_path} is not found")
    
    if os.path.isfile(gt_img_path):
        gt_img = tifffile.imread(gt_img_path)
        viewer.add_image(gt_img, name='gt_img', visible=False)


viewer = napari.Viewer()
viewer.window.add_dock_widget(NIDDL_widget, name= 'NIDDL Denoise', area='right')
napari.run()
    
