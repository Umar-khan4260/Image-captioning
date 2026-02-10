import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import nltk
import os
from utils.model_utils import ImageCaptioningModel
from utils.inference import greedy_search, beam_search
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# Set page config
st.set_page_config(
    page_title="Neural Storyteller - Image Captioning",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .caption-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 1rem 0;
    }
    .stButton button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vocabulary():
    """Load vocabulary from pickle file"""
    with open('models/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab

@st.cache_resource
def load_model():
    """Load pre-trained model"""
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary to get size
    vocab = load_vocabulary()
    
    # Initialize model
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embedding_dim=256,
        hidden_size=512,
        num_layers=1,
        dropout=0.5
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device, vocab

@st.cache_resource
def load_encoder():
    """Load ResNet50 encoder for feature extraction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    # Image preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return model, transform, device

def extract_features(image, encoder_model, transform, device):
    """Extract features from image using ResNet50"""
    img = image.convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = encoder_model(img_t).view(1, -1)
    
    return features

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Neural Storyteller - Image Captioning</h1>', unsafe_allow_html=True)
    st.markdown("""
    Generate descriptive captions for your images using deep learning.
    Upload an image and let the AI tell its story!
    """)
    
    # Initialize session state
    if 'caption' not in st.session_state:
        st.session_state.caption = ""
    if 'image' not in st.session_state:
        st.session_state.image = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Decoding method selection
        decoding_method = st.radio(
            "Select captioning method:",
            ["Greedy Search", "Beam Search"]
        )
        
        if decoding_method == "Beam Search":
            beam_width = st.slider("Beam width:", min_value=1, max_value=10, value=3)
        else:
            beam_width = 1
        
        max_length = st.slider("Maximum caption length:", min_value=10, max_value=50, value=20)
        
        st.divider()
        
        # Load models (cached)
        with st.spinner("Loading models..."):
            caption_model, caption_device, vocab = load_model()
            encoder_model, transform, encoder_device = load_encoder()
        
        st.success("✅ Models loaded successfully!")
        
        st.divider()
        
        # Information
        st.markdown("### 📚 About")
        st.markdown("""
        This model uses:
        - **Encoder**: ResNet50 (feature extraction)
        - **Decoder**: LSTM with attention
        - **Dataset**: Flickr30k
        - **Training**: 10 epochs with Adam optimizer
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for caption generation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.session_state.image = image
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary"):
                with st.spinner("Extracting image features..."):
                    features = extract_features(image, encoder_model, transform, encoder_device)
                
                with st.spinner("Generating caption..."):
                    if decoding_method == "Greedy Search":
                        caption = greedy_search(
                            caption_model, 
                            features, 
                            vocab, 
                            max_length=max_length, 
                            device=caption_device
                        )
                    else:
                        caption = beam_search(
                            caption_model, 
                            features, 
                            vocab, 
                            beam_width=beam_width,
                            max_length=max_length, 
                            device=caption_device
                        )
                    
                    st.session_state.caption = caption
                
                st.success("Caption generated successfully!")
    
    with col2:
        st.header("📝 Generated Caption")
        
        if st.session_state.caption:
            # Display caption with styling
            st.markdown(f'<div class="caption-box"><strong>Caption:</strong><br>{st.session_state.caption}</div>', 
                       unsafe_allow_html=True)
            
            # Copy to clipboard
            st.code(st.session_state.caption, language="text")
            
            # Download caption
            caption_text = st.session_state.caption
            st.download_button(
                label="📥 Download Caption",
                data=caption_text,
                file_name="generated_caption.txt",
                mime="text/plain"
            )
            
            # Try another button
            if st.button("🔄 Try Another Image"):
                st.session_state.caption = ""
                st.session_state.image = None
                st.rerun()
        else:
            st.info("👈 Upload an image and click 'Generate Caption' to see results here.")
            
            # Example captions
            st.markdown("### 💡 Example Captions")
            examples = [
                "A group of people riding bicycles on a city street",
                "A golden retriever playing with a ball in the park",
                "A beautiful sunset over mountains with clouds",
                "Two children laughing while eating ice cream"
            ]
            
            for example in examples:
                st.markdown(f"- {example}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #6B7280;'>
        <p>Built with ❤️ using PyTorch & Streamlit | Model: Encoder-Decoder with LSTM</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
