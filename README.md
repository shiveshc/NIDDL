# whole-brain_DeepDenoising
Deep denoising pushes the limit of functional data acquisition by recovering high SNR calcium traces from low SNR videos acquired using low laser power or smaller exposure time. Thus deep denoising enables faster and longer volumetric recordings.

# Installation
Installation steps tested for Windows10 and Python 3.5

### 1. Download whole-brain_DeepDenoising repository
Open Git Bash terminal, navigate to desired location and clone repository using `git clone https://github.com/shiveshc/whole-brain_DeepDenoising.git`

Or click on `Clone or download` button on top right corner and `Download ZIP`

### 2. Setting up venv and installing libraries
Open command line terminal as administrator and navigate to cloned repository path using `cd .\whole-brain_DeepDenoising`

Next run following commands-
1. `python -m venv env`
2. `env\Scripts\activate.bat`
3. `python -m pip install --upgrade "pip < 21.0"`
4. `pip install -r requirements.txt`

Installation should take 10-15 minutes.

Common installation errors -
1. pip version is not upgraded, 
    __solution :__ upgrade pip using `python -m pip install --upgrade pip`
2. pip version is not compatible with python version
    __solution :__ install suitable pip version
    

