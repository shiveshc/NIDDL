![GitHub repo size](https://img.shields.io/github/repo-size/shiveshc/NIDDL)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/shiveshc/NIDDL)
![Python version](https://img.shields.io/badge/python-v3.5-blue)
![visitors](https://visitor-badge.glitch.me/badge?page_id=shiveshc.NIDDL&left_color=green&right_color=red)
![GitHub](https://img.shields.io/github/license/shiveshc/NIDDL)
![Suggestions](https://img.shields.io/badge/suggestions-welcome-green)
<!-- ![GitHub all releases](https://img.shields.io/github/downloads/shiveshc/NIDDL/total) -->
![Image](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)


# NIDDL - Neuro Imaging Denoising Via Deep Learning
Deep denoising pushes the limit of functional data acquisition by recovering high SNR calcium traces from low SNR videos acquired using low laser power or smaller exposure time. Thus deep denoising enables faster and longer volumetric recordings. For more details, please check our [paper](https://www.nature.com/articles/s41467-022-32886-w).

If you find our work useful, please cite
```
Chaudhary, S., Moon, S. & Lu, H. Fast, efficient, and accurate neuro-imaging denoising via supervised deep learning. Nat Commun 13, 5165 (2022). https://doi.org/10.1038/s41467-022-32886-w
```

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

### 1. Download pytorch version of NIDDL repository
Clone pytorch branch of repository using `git clone -b pytorch https://github.com/shiveshc/NIDDL.git`.

### 2. Setting up env and installing dependencies
Open command line terminal as administrator and navigate to cloned repository path using `cd ~/NIDDL`.

Next run following commands - 
1. `conda env create -f environment.yml`
2. `conda activate niddl-env`

Installation should take ~10 minutes.

#### Troubleshooting
`environment.yml` has been tested on Linux. Installation on MacOS or Windows may throw errors. In this case you can try following -
1. `cd ~/NIDDL`
2. `conda create -n niddl-env python=3.9
3. `pip install ipykernel`
4. `jupyter notebook`
3. Run 1st cell in `example.ipynb`, manually install required libraries using `pip install <lib>`

### 3. Usage
1. Create jupyter kernel in `niddl-env` using `jupyter notebook`
2. Follow instructions in `example.ipynb`
    
# Additional system requirements
Model training using GPUs is much faster and thus preferred. To be able to use GPUs, suitable NVIDIA drivers and CUDA libraries must be installed. Please make sure your CUDA versions are compatible with Pytorch 2.0.1.

# Train on new dataset
1. Create jupyter kernel within `niddl-env`.
2. Follow instructions in `example.ipynb`.

3. Structure of training data folder should be organised as below -
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
