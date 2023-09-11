#!/bin/bash
#SBATCH --job-name=comp3710
#SBATCH --nodes=1
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

source /home/Student/s4583670/miniconda3/bin/activate comp3710
python3 resnet.py