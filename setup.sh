#!/bin/bash
# setup_project.sh

PROJECT_NAME="latent-bridge"
ENV_NAME="lb-env"

echo "🚀 Initializing $PROJECT_NAME..."

# 1. Create Directory Structure
mkdir -p $PROJECT_NAME/{notebooks,configs,data/{raw,processed},scripts,tests,src/{data,models,utils,pipeline}}
touch $PROJECT_NAME/src/__init__.py
touch $PROJECT_NAME/src/data/__init__.py
touch $PROJECT_NAME/src/models/__init__.py
touch $PROJECT_NAME/src/utils/__init__.py
touch $PROJECT_NAME/src/pipeline/__init__.py

# 2. Create Initial Configuration (Hydra style)
cat <<EOT > $PROJECT_NAME/configs/config.yaml
project_name: "latent-bridge"
device: "auto" # Will be handled by src/utils/hardware.py

experiment:
  name: "mlp_vs_cnn_features"
  batch_size: 64
  epochs: 10
  lr: 1e-3

model:
  type: "mlp_direct" # options: [mlp_direct, cnn_hybrid]
  input_dim: 784
  hidden_dims: [512, 256]
  num_classes: 10
EOT

# 3. Create Hardware Auto-Detection Utility
cat <<EOT > $PROJECT_NAME/src/utils/hardware.py
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
EOT

# 4. Create Requirements & Environment
cat <<EOT > $PROJECT_NAME/requirements.txt
torch==2.1.0
torchvision
numpy<2.0
pillow
opencv-python-headless
hydra-core
omegaconf
tqdm
pytest
EOT

# 5. Setup Conda
conda create -n $ENV_NAME python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
pip install -r $PROJECT_NAME/requirements.txt
pip install -e $PROJECT_NAME/ # Install src as an editable package

echo "✅ Structure built. To start: cd $PROJECT_NAME && conda activate $ENV_NAME"