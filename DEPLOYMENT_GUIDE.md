# 🚀 Complete Deployment Guide - Streamlit on GitHub

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step-by-Step Deployment](#step-by-step-deployment)
3. [Handling Large Model Files](#handling-large-model-files)
4. [Troubleshooting](#troubleshooting)
5. [Optional Enhancements](#optional-enhancements)

---

## Prerequisites

### What You Need:
- ✅ Trained model (`complete_model_package.pkl` from Kaggle)
- ✅ GitHub account
- ✅ Git installed on your computer
- ✅ All project files ready

### Download Model from Kaggle:
1. Go to your Kaggle notebook
2. Click **File** → **Download** → Download `complete_model_package.pkl`
3. Save it to your project folder

---

## Step-by-Step Deployment

### **STEP 1: Set Up Local Project**

#### 1.1 Create Project Folder
```bash
# On Windows
mkdir image-captioning
cd image-captioning

# On Mac/Linux
mkdir image-captioning
cd image-captioning
```

#### 1.2 Add All Files

Your folder structure should look like this:
```
image-captioning/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── complete_model_package.pkl
```

**Copy these files** (I've created them for you):
- `app.py` - Streamlit application
- `requirements.txt` - Dependencies
- `README.md` - Documentation
- `.gitignore` - Git ignore rules
- `complete_model_package.pkl` - Your trained model (download from Kaggle)

---

### **STEP 2: Test Locally** (Recommended)

Before deploying, test your app works:

#### 2.1 Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 2.2 Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2.3 Run Streamlit
```bash
streamlit run app.py
```

✅ **Success!** If the app opens in your browser (http://localhost:8501), you're ready to deploy.

---

### **STEP 3: Set Up GitHub Repository**

#### 3.1 Create GitHub Repository

1. Go to [github.com](https://github.com)
2. Click **"+"** → **"New repository"**
3. Fill in:
   - **Repository name**: `image-captioning`
   - **Description**: `AI-powered image captioning with Seq2Seq model`
   - **Public** or **Private** (your choice)
   - ❌ **Don't** initialize with README (we already have one)
4. Click **"Create repository"**

#### 3.2 Initialize Git Locally

```bash
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Image captioning app"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/image-captioning.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

### **STEP 4: Deploy to Streamlit Cloud**

#### 4.1 Go to Streamlit Cloud

1. Visit [share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your repositories

#### 4.2 Create New App

1. Click **"New app"** button

2. Fill in deployment settings:
   - **Repository**: Select `YOUR_USERNAME/image-captioning`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): Choose a custom name like `neural-storyteller`

3. Click **"Deploy!"**

#### 4.3 Wait for Deployment

- ⏱️ First deployment takes **3-5 minutes**
- 📊 You can watch the logs in real-time
- ✅ App will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## Handling Large Model Files

### Problem: GitHub Limits File Size to 100MB

If your `complete_model_package.pkl` is larger than 100MB, you have 3 options:

---

### **Option 1: Git LFS (Recommended)**

Git Large File Storage handles big files properly.

#### Install Git LFS

**Windows:**
```bash
# Download from https://git-lfs.github.com/
# Or use winget:
winget install GitHub.GitLFS
```

**Mac:**
```bash
brew install git-lfs
```

**Linux:**
```bash
sudo apt-get install git-lfs
```

#### Set Up Git LFS

```bash
# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.pth"

# Add .gitattributes
git add .gitattributes

# Add model file
git add complete_model_package.pkl

# Commit and push
git commit -m "Add model with Git LFS"
git push origin main
```

---

### **Option 2: Google Drive + Auto-Download**

Upload model to Google Drive and download in app.

#### 2.1 Upload to Google Drive

1. Upload `complete_model_package.pkl` to Google Drive
2. Right-click → **Get link** → Set to **"Anyone with the link"**
3. Copy the file ID from URL:
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view
   ```

#### 2.2 Modify `app.py`

Add this at the top of `app.py`:

```python
import gdown
import os

def download_model_if_needed():
    """Download model from Google Drive if not present"""
    model_file = 'complete_model_package.pkl'
    
    if not os.path.exists(model_file):
        st.info("Downloading model (one-time, ~2 minutes)...")
        
        # Replace with your file ID
        file_id = 'YOUR_FILE_ID_HERE'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        gdown.download(url, model_file, quiet=False)
        st.success("Model downloaded successfully!")

# Call before loading model
download_model_if_needed()
```

#### 2.3 Update `requirements.txt`

Add `gdown`:
```
torch==2.1.0
torchvision==0.16.0
streamlit==1.29.0
Pillow==10.1.0
numpy==1.24.3
gdown==4.7.1
```

#### 2.4 Remove from Git

```bash
# Add to .gitignore
echo "complete_model_package.pkl" >> .gitignore

# Remove from tracking
git rm --cached complete_model_package.pkl

# Commit
git add .gitignore
git commit -m "Use Google Drive for model file"
git push
```

---

### **Option 3: Hugging Face Hub**

Upload model to Hugging Face and download in app.

#### 3.1 Upload to Hugging Face

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new model repository
3. Upload `complete_model_package.pkl`

#### 3.2 Modify `app.py`

```python
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model_from_hf():
    """Download model from Hugging Face Hub"""
    model_path = hf_hub_download(
        repo_id="YOUR_USERNAME/image-captioning",
        filename="complete_model_package.pkl"
    )
    
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    return model_package
```

#### 3.3 Update `requirements.txt`

Add:
```
huggingface-hub==0.19.4
```

---

## Troubleshooting

### Issue 1: "Module not found" Error

**Solution:**
```bash
# Make sure requirements.txt has all dependencies
pip freeze > requirements.txt

# Push updated requirements
git add requirements.txt
git commit -m "Update requirements"
git push
```

### Issue 2: Out of Memory on Streamlit Cloud

Streamlit Cloud has **1GB RAM limit** for free tier.

**Solutions:**
- Use CPU-only PyTorch: `torch==2.1.0+cpu`
- Reduce model size
- Upgrade to Streamlit Cloud paid tier

### Issue 3: Slow App Loading

**Solutions:**
1. Use `@st.cache_resource` for model loading (already in `app.py`)
2. Reduce model size
3. Use model compression (quantization)

### Issue 4: App Crashes on Image Upload

**Check:**
- Image preprocessing matches training
- Device compatibility (CPU vs GPU)
- PIL image conversion

**Debug:**
```python
# Add error handling
try:
    caption = beam_search(model, features, vocab, beam_width=beam_width, device=device)
    st.success(f"Caption: {caption}")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.error(f"Details: {type(e).__name__}")
```

---

## Optional Enhancements

### Add Examples to Sidebar

```python
# In app.py, add to sidebar
st.sidebar.markdown("### Try These Examples")

example_images = {
    "Dog in park": "examples/dog.jpg",
    "Beach sunset": "examples/beach.jpg",
}

for name, path in example_images.items():
    if st.sidebar.button(name):
        st.session_state['example_image'] = path
```

### Add Analytics

```python
# Track usage with simple counter
import json

def log_usage():
    try:
        with open('usage.json', 'r') as f:
            data = json.load(f)
    except:
        data = {'count': 0}
    
    data['count'] += 1
    
    with open('usage.json', 'w') as f:
        json.dump(data, f)

# Call after generating caption
log_usage()
```

### Add Image History

```python
# Store recent captions
if 'history' not in st.session_state:
    st.session_state.history = []

# After generating
st.session_state.history.append({
    'image': uploaded_file,
    'caption': caption,
    'timestamp': datetime.now()
})

# Display in sidebar
st.sidebar.markdown("### Recent Captions")
for item in st.session_state.history[-5:]:
    st.sidebar.write(item['caption'])
```

---

## 🎉 You're Done!

Your app should now be live at: `https://YOUR_APP_NAME.streamlit.app`

### Share Your App:
- 📱 **Direct Link**: Send the URL to anyone
- 🐦 **Twitter**: Share with #Streamlit #ImageCaptioning
- 💼 **LinkedIn**: Add to your portfolio
- 📧 **Email**: Include in your resume/CV

### Next Steps:
1. ⭐ Star your repo on GitHub
2. 📝 Add screenshots to README
3. 🎨 Customize the UI
4. 📊 Add more evaluation metrics
5. 🚀 Train a better model with attention

---

## 📞 Need Help?

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Create an issue in your repo
- **Discord**: Join Streamlit Discord

---

**Good luck with your deployment! 🚀**
