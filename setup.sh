#!/bin/bash

ENV_NAME="MatGen"

echo "Creating conda env $ENV_NAME..."
conda create -n $ENV_NAME python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
echo "Now in $CONDA_PREFIX"

echo "Installing PyTorch + CUDA Toolkit..."
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 cuda-toolkit=11.7 cuda-nvcc=11.7 -c pytorch -c nvidia -y

echo "Installing pytorch3d dependencies..."
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y

echo "Installing pytorch3d..."
conda install pytorch3d -c pytorch3d -y

conda install xformers -c xformers -y

echo "Installing tinycudann from source (may take a while)..."
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation

echo "Installing flash-attention (may take a while)..."
MAX_JOBS=1 pip -v install flash-attn==2.7.2.post1 --no-build-isolation

echo "Installing remaining dependencies..."
pip install \
    trimesh scikit-learn opencv-python matplotlib imageio \
    diffusers==0.21.0 einops transformers==4.40.2 open-clip-torch \
    gradio==3.48.0 pytorch-lightning==1.9.1 omegaconf triton accelerate \
    plotly xatlas wandb point_cloud_utils rtree cython packaging \
    blenderproc dreifus torch-fidelity clean-fid huggingface-hub==0.25.0 \
    lovely-tensors numpy==1.26.4

echo "Installation Complete!"