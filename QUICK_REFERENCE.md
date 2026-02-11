# 🚀 Quick Reference Card - Streamlit Deployment

## 📦 Files You Need

```
image-captioning/
├── app.py                          ← Streamlit app (provided)
├── requirements.txt                ← Dependencies (provided)
├── README.md                       ← Documentation (provided)
├── .gitignore                      ← Git rules (provided)
├── complete_model_package.pkl      ← YOUR MODEL (from Kaggle)
└── DEPLOYMENT_GUIDE.md             ← Instructions (provided)
```

---

## ⚡ Quick Commands

### Local Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### GitHub Setup
```bash
# Initialize Git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/image-captioning.git

# Push
git branch -M main
git push -u origin main
```

### Git LFS (for large model files)
```bash
# Install Git LFS
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Linux: sudo apt-get install git-lfs

# Initialize
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.pth"

# Add and commit
git add .gitattributes
git add complete_model_package.pkl
git commit -m "Add model with LFS"
git push
```

---

## 🌐 Streamlit Cloud Deployment

### Steps:
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - Repository: `YOUR_USERNAME/image-captioning`
   - Branch: `main`
   - Main file: `app.py`
5. Click **"Deploy!"**
6. Wait 3-5 minutes
7. Your app is live! 🎉

### Your App URL:
```
https://YOUR_APP_NAME.streamlit.app
```

---

## 🔧 Common Issues

### Issue 1: File too large for GitHub
**Solution:** Use Git LFS (see commands above)

### Issue 2: Module not found
```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

### Issue 3: Out of memory on Streamlit
**Solution:** Already handled in `app.py` with caching

### Issue 4: App won't start
- Check all files are pushed to GitHub
- Check model file is present
- Check logs on Streamlit Cloud

---

## 📱 Share Your App

### Direct Link
```
https://YOUR_APP_NAME.streamlit.app
```

### Embed in Website
```html
<iframe src="https://YOUR_APP_NAME.streamlit.app" width="100%" height="600"></iframe>
```

### QR Code
Generate at: https://www.qr-code-generator.com/
Enter your app URL

---

## 🎯 Quick Checklist

- [ ] All files in folder
- [ ] Model downloaded from Kaggle
- [ ] Tested locally with `streamlit run app.py`
- [ ] GitHub repo created
- [ ] Code pushed to GitHub
- [ ] Streamlit app deployed
- [ ] App tested online
- [ ] URL shared!

---

## 📞 Help & Resources

- **Streamlit Docs**: docs.streamlit.io
- **Forum**: discuss.streamlit.io  
- **Full Guide**: See DEPLOYMENT_GUIDE.md
- **Checklist**: See DEPLOYMENT_CHECKLIST.md

---

## 🎓 Your Achievement

You've built and deployed:
- ✅ Deep learning model
- ✅ Web application
- ✅ Cloud deployment
- ✅ Public portfolio piece

**Congratulations! 🎉**

---

**Pro Tip:** Add this to your resume/CV under "Projects":
```
Image Captioning Web App | Python, PyTorch, Streamlit
• Trained Seq2Seq model on 31K images using ResNet50 + LSTM
• Deployed production-ready ML application on Streamlit Cloud  
• Live demo: https://YOUR_APP_NAME.streamlit.app
```
