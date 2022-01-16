# whole-brain_DeepDenoising
Deep denoising pushes the limit of functional data acquisition by recovering high SNR calcium traces from low SNR videos acquired using low laser power or smaller exposure time. Thus deep denoising enables faster and longer volumetric recordings.

# Installation
Installation steps tested for Windows 10 64-bit and Python 3.5

### 1. Download whole-brain_DeepDenoising repository
Open Git Bash terminal, navigate to desired location and clone repository using `git clone https://github.com/shiveshc/whole-brain_DeepDenoising.git`

Or click on `Code` button on top right corner and `Download ZIP`

### 2. Setting up venv and installing libraries
Open command line terminal as administrator and navigate to cloned repository path using `cd .\whole-brain_DeepDenoising`

Next run following commands-
1. `python -m venv env`
2. `env\Scripts\activate.bat`
3. `python -m pip install --upgrade "pip < 21.0"`
4. `pip install -r requirements.txt`

Installation should take 10-15 minutes.

Common installation errors -
1. pip version is not upgraded\
    __solution :__ upgrade pip using `python -m pip install --upgrade pip`
2. pip version is not compatible with python version\
    __solution :__ install suitable pip version e.g. pip version < 21.0 is compatible with Python 3.5.
    
# Additional system requirements
Model training using GPUs is much faster and thus preferred. To be able to use GPUs, suitable NVIDIA drivers and CUDA libraries must be installed. Additional instructions on setting up tensorflow-gpu can be found at <a href = "https://www.tensorflow.org/install/gpu#Software_requirements">Software requirements</a> and <a href = "https://www.tensorflow.org/install/gpu#windows_setup">Windows setup</a>.

For tensorflow 1.6.0 currently setup in venv, CUDA v9.0 and `cudnn-9.0-windows10-x64-v7.6.4.38` downloaded from <a href= "https://developer.nvidia.com/rdp/cudnn-archive">cudnn archives</a> works for Windows 10 64-bit.

# Train on new dataset
1. To train network on new data, pairs of noisy i.e. low SNR images (acquired at low laser power or small exposure time conditions) and clean i.e. High SNR images (acquired at high laser power or long exposure time conditions) are needed. Currently supported image size is 512 x 512 x d where d is number of images in stack. For other image sizes, either resize images first or channels dimensions need to be changed in architecture files in `cnn_archs` folder.

2. Structure of training data folder should be organised as provided below -
```
data
├───gt_imgs
│   ├───img_1
│   │       z_1.tif
│   │       z_2.tif
│   │       ...
│   │
│   ├───img_2
│   │       z_1.tif
│   │       z_2.tif
│   │       ...
│   │
│   └───img_3
│           z_1.tif
│           z_2.tif
│           ...
│
└───noisy_imgs
    ├───img_1
    │       z_1.tif
    │       z_2.tif
    │       ...
    │
    ├───img_2
    │       z_1.tif
    │       z_2.tif
    │       ...
    │
    └───img_3
            z_1.tif
            z_2.tif
            ...
 ```
 
 3. Run `python train.py -h` to see usage and input arguments. The output on terminal should look like
 ```
usage: train.py [-h] [-out OUT]
                [-arch {unet,unet_fixed,hourglass_wres,hourglass_wores}]
                [-mode {2D,2.5D,3D}] [-depth DEPTH] [-loss {l2,l1}]
                [-epochs EPOCHS] [-lr LR] [-bs BS] [-tsize TSIZE]
                data run

train CNN model to denoise volumetric functional recordings

positional arguments:
  data                  training data path
  run                   run number to distinguish different runs

optional arguments:
  -h, --help            show this help message and exit
  -out OUT              location for saving results
  -arch {unet,unet_fixed,hourglass_wres,hourglass_wores}
                        CNN architechture to use for training (default is
                        hourglass_wres)
  -mode {2D,2.5D,3D}    training mode (default is 2D)
  -depth DEPTH          stack depth to use for training (must be odd number,
                        default is 1)
  -loss {l2,l1}         L2 or L1 loss for training (default is l1)
  -epochs EPOCHS        number of epochs to train the model for (150-200 is
                        good choice, default is 150)
  -lr LR                learning rate (default is 0.001)
  -bs BS                batch size of training (default is 6)
  -tsize TSIZE          data size (number of images) to use for training
  ```
  e.g. to train 
