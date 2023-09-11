#!/bin/bash

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy
conda install matplotlib
conda install pandas
conda install scikit-learn