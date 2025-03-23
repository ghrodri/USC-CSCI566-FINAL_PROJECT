#!/bin/bash

# Name of the conda environment
ENV_NAME="CSCI566_ENV"

echo "Creating conda environment: $ENV_NAME with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118

echo "Installing additional packages..."
pip install \
  transformers==4.39.1 \
  pandas==2.2.1 \
  scikit-learn==1.4.1 \
  numpy==1.26.4 \
  tqdm==4.66.2 \
  matplotlib==3.8.3 \
  Pillow==10.2.0 \
  opencv-python==4.9.0.80

echo "Verifying CUDA availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"

echo "Environment $ENV_NAME is ready."

