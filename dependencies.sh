#!/bin/bash

# Usage: pip install -r requirements.txt

# Dowload the yolov7 repository
git clone https://github.com/WongKinYiu/yolov7.git
sudo apt-get update
sudo apt-get install ffmpeg

# Check if pip is installed, if not, try to install it
if ! command -v pip &> /dev/null
then
    echo "pip could not be found, attempting to install"
    sudo apt install python3-pip
fi

# Check if virtualenv is installed, if not, try to install it
if ! command -v virtualenv &> /dev/null
then
    echo "virtualenv could not be found, attempting to install"
    pip install virtualenv
fi

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

# Install the python dependencies
pip install -r yolov7/requirements.txt

pip install pytube

pip install opencv-python

pip install pytransform3d



