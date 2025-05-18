#!/bin/bash

set -e

echo "Checking for NVIDIA GPU and drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Installing via ubuntu-drivers..."
    sudo apt update
    sudo ubuntu-drivers autoinstall
    echo "Reboot required to activate drivers. Run this script again after reboot."
    exit 0
else
    echo "NVIDIA GPU detected:"
    nvidia-smi
fi

echo "Checking for Python 3.10..."
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10 and pip..."
    sudo apt install -y python3.10 python3.10-venv python3.10-dev python3-pip
fi

echo "Cleaning up any CPU-only Torch versions..."
python3.10 -m pip uninstall -y torch torchvision

echo "Installing Torch 2.0.1 + CUDA 11.8..."
python3.10 -m pip install --upgrade pip
python3.10 -m pip install \
  torch==2.0.1+cu118 \
  torchvision==0.15.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

echo "Verifying CUDA availability in PyTorch..."
python3.10 -c "import torch; print('CUDA Available:' if torch.cuda.is_available() else 'CUDA NOT Available')"

echo "Setup complete. You can now run your RIFE model using GPU acceleration."
