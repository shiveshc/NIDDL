#!/usr/bin/bash

#SBATCH --job-name=unet_fixed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_geforce_gtx_1080_ti:1
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --output=/home/schaudhary/whole-brain_DeepDenoising/slurm_jobs/unet_fixed-%j.out

source ~/.bashrc
conda activate pytorch-env
python3 /home/schaudhary/whole-brain_DeepDenoising/train.py\
    -data /home/schaudhary/whole-brain_DeepDenoising/data/synthetic_data/signal_5\
    -epochs 30\
    -arch unet_fixed\
    -mode 2D\
    -depth 1\
    -run 1\
    -out /home/schaudhary/whole-brain_DeepDenoising/test_runs