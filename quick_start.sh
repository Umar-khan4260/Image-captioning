#!/bin/bash

# Quick Start Script for Image Captioning Deployment
# For Mac/Linux Users

echo "================================================"
echo "Image Captioning - Quick Start"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed!"
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed!"
    echo "Please install Git from https://git-scm.com/downloads"
    exit 1
fi

echo "✓ Python and Git are installed"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Place your model file (complete_model_package.pkl) in this folder"
echo "2. Run: streamlit run app.py"
echo "3. Your app will open in the browser"
echo ""
echo "To deploy to Streamlit Cloud:"
echo "1. Create GitHub repo"
echo "2. Push your code: git init && git add . && git commit -m 'Initial commit' && git push"
echo "3. Go to share.streamlit.io and deploy"
echo ""
echo "See DEPLOYMENT_GUIDE.md for detailed instructions"
echo ""
