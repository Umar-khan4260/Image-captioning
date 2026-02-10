# simple_app.py - Minimal test app
import streamlit as st
import sys
import os

st.set_page_config(page_title="Test PyTorch", layout="wide")

st.title("🧪 PyTorch Installation Test")

# Show Python info
st.subheader("Python Information")
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir('.')}")

# Try importing PyTorch
try:
    import torch
    st.success("✅ PyTorch imported successfully!")
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test a simple tensor operation
    x = torch.rand(2, 3)
    st.write(f"Random tensor shape: {x.shape}")
    st.write(f"Tensor:\n{x}")
    
except Exception as e:
    st.error(f"❌ PyTorch import failed: {str(e)}")
    st.write("Trying alternative import method...")
    
    # Try CPU-only version
    try:
        import subprocess
        import sys
        
        st.info("Attempting to install PyTorch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "--no-cache-dir"])
        
        import torch
        st.success("✅ PyTorch installed and imported!")
        st.write(f"PyTorch version: {torch.__version__}")
        
    except Exception as e2:
        st.error(f"❌ Installation also failed: {str(e2)}")

# Show requirements
if os.path.exists("requirements.txt"):
    st.subheader("Requirements.txt contents:")
    with open("requirements.txt", "r") as f:
        st.code(f.read())
