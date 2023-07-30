![GitHub repo size](https://img.shields.io/github/repo-size/shiveshc/whole-brain_DeepDenoising)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/shiveshc/whole-brain_DeepDenoising)
![Python version](https://img.shields.io/badge/python-v3.5-blue)
![visitors](https://visitor-badge.glitch.me/badge?page_id=shiveshc.whole-brain_DeepDenoising&left_color=green&right_color=red)
![GitHub](https://img.shields.io/github/license/shiveshc/whole-brain_DeepDenoising)
![Suggestions](https://img.shields.io/badge/suggestions-welcome-green)
<!-- ![GitHub all releases](https://img.shields.io/github/downloads/shiveshc/whole-brain_DeepDenoising/total) -->


# whole-brain_DeepDenoising
Deep denoising pushes the limit of functional data acquisition by recovering high SNR calcium traces from low SNR videos acquired using low laser power or smaller exposure time. Thus deep denoising enables faster and longer volumetric recordings. For more details, please check our [paper](https://www.nature.com/articles/s41467-022-32886-w).

<p align = "center"><b>Denoise whole-brain videos</b></p>
<img src = "extra/wb_denoising.gif" width=100% align="center">

<p align = "center"><b>Denoise mechanosensory neurites</b></p>
<img src = "extra/neurite_denoising.gif" width=100% align="center">

<p align = "center"><b>Recover high SNR calcium traces</b></p>
<img src = "extra/denoise_traces.png" width=100% align="center">

# Contents
1. [Installation](#installation)
2. [Additional system requirements](#additional-system-requirements)
3. [Train on new dataset](#train-on-new-dataset)
4. [Denoise calcium activity recordings](#denoise-calcium-activity-recordings)

# Installation
Installation steps tested for Linux and Python 3.9

### 1. Download pytorch version of whole-brain_DeepDenoising repository
Clone pytorch branch of repository using `git clone -b pytorch https://github.com/shiveshc/whole-brain_DeepDenoising.git`.

### 2. Setting up venv and installing libraries
Open command line terminal as administrator and navigate to cloned repository path using `cd /whole-brain_DeepDenoising`.

Next run following commands - 
1. `conda env create -f environment.yml`

Installation should take ~10 minutes.
    
# Additional system requirements
Model training using GPUs is much faster and thus preferred. To be able to use GPUs, suitable NVIDIA drivers and CUDA libraries must be installed. Additional instructions on setting up tensorflow-gpu can be found at <a href = "https://www.tensorflow.org/install/gpu#Software_requirements">Software requirements</a> and <a href = "https://www.tensorflow.org/install/gpu#windows_setup">Windows setup</a>.

For tensorflow 1.6.0 currently setup in venv, CUDA v9.0 and `cudnn-9.0-windows10-x64-v7.6.4.38` downloaded from <a href= "https://developer.nvidia.com/rdp/cudnn-archive">cudnn archives</a> works for Windows 10 64-bit.

# Train on new dataset
1. Follow instructions in `example.ipynb`.

2. Structure of training data folder should be organised as below -
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

# Denoise calcium activity recordings
1. Structure of functional recording video datasets should be organised as below - 
    ```
    vid1
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
        ├───img_3
        │       z_1.tif
        │       z_2.tif
        │       ...
        ...
    ```
    Here `img_1`, `img_2` etc. can correspond to individual time-points in videos.

2. Follow instruction in `example.ipynb`. 