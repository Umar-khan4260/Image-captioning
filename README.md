# 🖼️ Neural Storyteller - Image Captioning

An AI-powered image captioning application that generates natural language descriptions for images using a Seq2Seq model with LSTM.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **AI-Powered Captioning**: Generate human-like descriptions for any image
- **Multiple Search Methods**: Choose between Greedy Search and Beam Search
- **Interactive Web Interface**: User-friendly Streamlit app
- **Real-time Processing**: Get captions in seconds
- **Pre-trained Model**: Trained on Flickr30k dataset with 31,000+ images

## 🏗️ Architecture

- **Encoder**: ResNet50 (pre-trained on ImageNet)
  - Extracts 2048-dimensional feature vectors from images
  
- **Decoder**: LSTM-based sequence generator
  - Embedding dimension: 256
  - Hidden size: 512
  - Vocabulary: ~8,000 words

## 📊 Model Performance

- **BLEU-4**: [Your score here]
- **METEOR**: [Your score here]
- **Training**: 10 epochs on Flickr30k dataset
- **Validation Loss**: [Your score here]

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/image-captioning.git
cd image-captioning
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model files**

Place the following files in the project root directory:
- `complete_model_package.pkl` (trained model and vocabulary)

> **Note**: The model file is too large for GitHub. Download it from [Google Drive link] or train your own using the Jupyter notebook.

### Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository

1. Make sure you have:
   - `app.py`
   - `requirements.txt`
   - `complete_model_package.pkl`
   - `.gitignore`
   - `README.md`

2. Create a GitHub repository and push your code

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)

2. Sign in with your GitHub account

3. Click **"New app"**

4. Fill in the details:
   - **Repository**: `YOUR_USERNAME/image-captioning`
   - **Branch**: `main`
   - **Main file path**: `app.py`

5. Click **"Deploy"**

6. Wait for deployment (2-5 minutes)

7. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

### Handling Large Model Files

If your model file is too large for GitHub (>100MB), use one of these options:

#### Option 1: Git LFS (Recommended)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.pth"

# Add and commit
git add .gitattributes
git add complete_model_package.pkl
git commit -m "Add model files with LFS"
git push
```

#### Option 2: External Storage
1. Upload model to Google Drive/Dropbox
2. Get a direct download link
3. Modify `app.py` to download the model on first run:

```python
import gdown
import os

@st.cache_resource
def download_model():
    if not os.path.exists('complete_model_package.pkl'):
        url = 'YOUR_GOOGLE_DRIVE_LINK'
        output = 'complete_model_package.pkl'
        gdown.download(url, output, quiet=False)
```

## 📁 Project Structure

```
image-captioning/
│
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── complete_model_package.pkl      # Trained model (not in repo)
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
│
├── notebooks/                      # Training notebooks (optional)
│   └── image-captioning.ipynb
│
└── assets/                         # Images for README (optional)
    └── demo.gif
```

## 🎯 Usage

1. **Upload an Image**: Click "Choose an image..." and select a JPG/PNG file

2. **Select Search Method**:
   - **Beam Search**: Better quality captions (recommended)
   - **Greedy Search**: Faster but may be less accurate

3. **Adjust Settings** (Beam Search only):
   - Beam Width: 1-5 (higher = better quality, slower)

4. **Generate Caption**: Click "Generate Caption" button

5. **View Result**: See the AI-generated description below your image

## 🖼️ Example Results

| Image | Generated Caption |
|-------|-------------------|
| ![Example 1](assets/example1.jpg) | "a dog running in a park" |
| ![Example 2](assets/example2.jpg) | "two people walking on a beach" |

## 🛠️ Technical Details

### Model Training

The model was trained on:
- **Dataset**: Flickr30k (31,783 images, 5 captions each)
- **Epochs**: 10
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Hardware**: NVIDIA Tesla T4 GPU

### Inference

- **Feature Extraction**: ResNet50 extracts image features
- **Caption Generation**: LSTM decoder generates words sequentially
- **Beam Search**: Explores multiple caption possibilities for better results

## 📚 Dataset

This project uses the [Flickr30k dataset](http://shannon.cs.illinois.edu/DenotationGraph/):
- 31,783 images
- 5 captions per image
- 158,915 total captions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Flickr30k Dataset**: [Young et al., 2014](http://shannon.cs.illinois.edu/DenotationGraph/)
- **ResNet50**: [He et al., 2015](https://arxiv.org/abs/1512.03385)
- **PyTorch**: Deep learning framework
- **Streamlit**: Web app framework

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/YOUR_USERNAME/image-captioning](https://github.com/YOUR_USERNAME/image-captioning)

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@misc{image-captioning-2024,
  author = {Your Name},
  title = {Neural Storyteller: Image Captioning with Seq2Seq},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/image-captioning}
}
```

---

<div align="center">
  Made with ❤️ using PyTorch and Streamlit
</div>
