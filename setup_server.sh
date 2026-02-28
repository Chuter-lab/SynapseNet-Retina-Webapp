#!/bin/bash
# Setup script for SynapseNet-Retina webapp on node02
# Run as root: echo '<pw>' | sudo -S bash setup_server.sh

set -e

# Create conda environment
echo "Creating conda environment..."
source /data/software/miniconda3/etc/profile.d/conda.sh
conda create -n synapsenet python=3.11 -y
conda activate synapsenet

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install gradio torch-em numpy opencv-python-headless tifffile scikit-image Pillow matplotlib

# Create directories
mkdir -p /data/thiru/models
mkdir -p /data/thiru/temp/uploads
mkdir -p /data/thiru/logs
mkdir -p /data/thiru/webapp

echo "Environment setup complete."
echo "Next: copy model checkpoints to /data/thiru/models/"
echo "  scp node01:/mnt/Projects/SynapseNet-Retina/models-env/finetuned/checkpoints/protected/retina_vesicles_best_retrain.pt /data/thiru/models/retina_vesicles_best.pt"
echo "  scp node01:/mnt/Projects/SynapseNet-Retina/models-env/finetuned/checkpoints/protected/retina_mitochondria_best_run008.pt /data/thiru/models/retina_mitochondria_best.pt"
echo "  scp node01:/mnt/Projects/SynapseNet-Retina/models-env/finetuned/checkpoints/protected/retina_membrane_best_run008.pt /data/thiru/models/retina_membrane_best.pt"
