#!/bin/bash
# setup.sh

echo "=== Setting up Image Captioning App ==="

# Install Python packages
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

echo "=== Setup Complete ==="
