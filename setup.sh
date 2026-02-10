#!/bin/bash
# setup.sh

echo "Setting up environment for Image Captioning App..."

# Update pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Create cache directory for PyTorch
mkdir -p ~/.cache/torch/hub/checkpoints

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

echo "Setup complete!"
