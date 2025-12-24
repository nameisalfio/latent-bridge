#!/bin/bash

# Define the root directory name
REPO_NAME="latent-bridge"

echo "🏗️ Building MLOps structure for $REPO_NAME..."

# Create core directory structure
mkdir -p configs/{model,dataset,trainer}
mkdir -p data/{raw,processed}
mkdir -p notebooks
mkdir -p scripts
mkdir -p src/{data,models,utils,pipeline}
mkdir -p tests

# Create placeholders for data and models to ensure git tracks the folders
touch data/.gitkeep
touch models/.gitkeep

# Initialize __init__.py files for the src package
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch src/pipeline/__init__.py

# Create a high-level Device Manager in utils
cat <<EOF > src/utils/device.py
import torch

def get_device() -> torch.device:
    """Auto-detects hardware: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def log_hardware():
    device = get_device()
    print(f"🚀 Execution Device: {device}")
EOF

# Create a placeholder for the Base Model interface
cat <<EOF > src/models/base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass
EOF

# Create a skeleton requirements.txt
cat <<EOF > requirements.txt
torch==2.1.0
torchvision
numpy<2.0
hydra-core
omegaconf
pydantic
opencv-python-headless
Pillow
tqdm
pytest
EOF

# Create a basic Dockerfile
cat <<EOF > Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "scripts/train.py"]
EOF

# Create a basic docker-compose.yaml
cat <<EOF > docker-compose.yaml
version: '3.8'
services:
  train:
    build: .
    volumes:
      - .:/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF

chmod +x setup.sh
echo "✅ Project structure complete."
echo "💡 Reminder: Run 'pip install -e .' to make the 'src' folder importable."