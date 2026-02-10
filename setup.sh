#!/bin/bash
# setup.sh - Setup script for Streamlit Cloud

echo "Setting up Image Captioning App..."

# Create necessary directories
mkdir -p ~/.cache/torch/hub/checkpoints
mkdir -p models

# Download ResNet50 weights if not present
if [ ! -f ~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth ]; then
    echo "Downloading ResNet50 weights..."
    wget -q -P ~/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/resnet50-11ad3fa6.pth
fi

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

# Check for model files
if [ ! -f models/best_model.pth ]; then
    echo "Warning: best_model.pth not found in models/ directory"
    echo "You need to upload your trained model file."
fi

if [ ! -f models/vocab.pkl ]; then
    echo "Warning: vocab.pkl not found in models/ directory"
    echo "You need to upload your vocabulary file."
fi

echo "Setup complete!"
