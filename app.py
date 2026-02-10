# app.py - Complete Image Captioning Streamlit App

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import nltk
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add utils directory to path
sys.path.append(str(Path(__file__).parent / "utils"))

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

# Import custom modules
from model_utils import ImageCaptioningModel
from inference import greedy_search, beam_search
from preprocessing import Vocabulary

# Set page config
st.set_page_config(
    page_title="Neural Storyteller - Image Captioning",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .caption-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3B82F6;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .caption-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        font-weight: bold;
        width: 100%;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.5);
    }
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #60A5FA;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #F59E0B;
    }
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #3B82F6, transparent);
        margin: 2rem 0;
    }
    .example-caption {
        background: white;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #10B981;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_models():
    """Load all models and return them cached"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    vocab_path = "models/vocab.pkl"
    if not os.path.exists(vocab_path):
        st.error(f"Vocabulary file not found at {vocab_path}")
        return None, None, None, None
    
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading vocabulary: {str(e)}")
        return None, None, None, None
    
    # Load caption model
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None, None, None, None
    
    try:
        # Initialize model
        caption_model = ImageCaptioningModel(
            vocab_size=len(vocab),
            embedding_dim=256,
            hidden_size=512,
            num_layers=1,
            dropout=0.5
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        caption_model.load_state_dict(checkpoint['model_state_dict'])
        caption_model.eval()
    except Exception as e:
        st.error(f"Error loading caption model: {str(e)}")
        return None, None, None, None
    
    # Load ResNet50 encoder for feature extraction
    try:
        encoder_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        encoder_model = nn.Sequential(*list(encoder_model.children())[:-1])
        encoder_model = encoder_model.to(device)
        encoder_model.eval()
    except Exception as e:
        st.error(f"Error loading ResNet50: {str(e)}")
        return None, None, None, None
    
    # Image preprocessing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    return caption_model, encoder_model, transform, vocab, device

def extract_image_features(image, encoder_model, transform, device):
    """Extract features from image using ResNet50"""
    try:
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        img_t = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = encoder_model(img_t)
            features = features.view(features.size(0), -1)  # Flatten to [1, 2048]
        
        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def generate_caption(features, caption_model, vocab, decoding_method, beam_width, max_length, device):
    """Generate caption using specified method"""
    try:
        if decoding_method == "Greedy Search":
            caption = greedy_search(
                caption_model, 
                features, 
                vocab, 
                max_length=max_length, 
                device=device
            )
        else:  # Beam Search
            caption = beam_search(
                caption_model, 
                features, 
                vocab, 
                beam_width=beam_width,
                max_length=max_length, 
                device=device
            )
        return caption
    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return None

def validate_vocabulary(vocab):
    """Validate that vocabulary has all required attributes"""
    required_attrs = ['word2idx', 'idx2word', 'start_token', 'end_token', 'pad_token', 'unk_token']
    missing = [attr for attr in required_attrs if not hasattr(vocab, attr)]
    
    if missing:
        st.error(f"Vocabulary missing attributes: {', '.join(missing)}")
        return False
    
    # Check if special tokens are in word2idx
    special_tokens = [vocab.start_token, vocab.end_token, vocab.pad_token, vocab.unk_token]
    for token in special_tokens:
        if token not in vocab.word2idx:
            st.error(f"Special token '{token}' not found in vocabulary")
            return False
    
    return True

def display_model_info(caption_model, vocab, device):
    """Display model information in sidebar"""
    with st.sidebar:
        st.markdown("### 📊 Model Info")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Device", "GPU" if device.type == "cuda" else "CPU")
        with col2:
            st.metric("Vocabulary", f"{len(vocab):,}")
        
        # Model parameters
        total_params = sum(p.numel() for p in caption_model.parameters())
        trainable_params = sum(p.numel() for p in caption_model.parameters() if p.requires_grad)
        
        st.markdown(f"""
        <div class="stat-box">
            <h4>Model Parameters</h4>
            <p>Total: {total_params:,}</p>
            <p>Trainable: {trainable_params:,}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Neural Storyteller - Image Captioning</h1>', unsafe_allow_html=True)
    st.markdown("""
    Generate descriptive captions for your images using deep learning. 
    Upload an image and let the AI tell its story!
    """)
    
    # Initialize session state
    if 'generated_caption' not in st.session_state:
        st.session_state.generated_caption = ""
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = 0
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Decoding method selection
        decoding_method = st.radio(
            "**Select captioning method:**",
            ["Greedy Search", "Beam Search"],
            help="Greedy: Faster, selects most likely word each step. Beam: Slower but often better quality."
        )
        
        # Beam width slider (only for beam search)
        if decoding_method == "Beam Search":
            beam_width = st.slider(
                "**Beam width:**",
                min_value=1,
                max_value=10,
                value=3,
                help="Higher values explore more possibilities but are slower"
            )
        else:
            beam_width = 1
        
        # Max length slider
        max_length = st.slider(
            "**Maximum caption length:**",
            min_value=10,
            max_value=50,
            value=20,
            help="Maximum number of words in generated caption"
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Load models
        with st.spinner("🔄 Loading models..."):
            caption_model, encoder_model, transform, vocab, device = load_models()
        
        if caption_model and encoder_model and vocab:
            st.markdown('<div class="success-box">✅ Models loaded successfully!</div>', unsafe_allow_html=True)
            
            # Validate vocabulary
            if not validate_vocabulary(vocab):
                st.stop()
            
            # Display model info
            display_model_info(caption_model, vocab, device)
        else:
            st.markdown('<div class="warning-box">⚠️ Models failed to load. Check console for details.</div>', unsafe_allow_html=True)
            st.stop()
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Information section
        st.markdown("### 📚 About")
        st.markdown("""
        <div class="info-box">
        <strong>Architecture:</strong><br>
        • Encoder: ResNet50<br>
        • Decoder: LSTM<br>
        • Dataset: Flickr30k<br>
        • Training: 10 epochs
        </div>
        """, unsafe_allow_html=True)
        
        # Example images section
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Use clear, well-lit images
        - Center the main subject
        - Beam search often produces better results
        - Adjust length based on complexity
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Supported formats: JPG, PNG, BMP, WEBP"
        )
        
        if uploaded_file is not None:
            try:
                # Open and display image
                image = Image.open(uploaded_file)
                st.session_state.uploaded_image = image
                
                # Display image
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Image info
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Format", image.format)
                with col_info2:
                    st.metric("Size", f"{image.size[0]}×{image.size[1]}")
                with col_info3:
                    st.metric("Mode", image.mode)
                
                # Generate caption button
                if st.button("🎯 Generate Caption", type="primary", use_container_width=True):
                    with st.spinner("Extracting image features..."):
                        features = extract_image_features(image, encoder_model, transform, device)
                    
                    if features is not None:
                        with st.spinner("Generating caption..."):
                            import time
                            start_time = time.time()
                            
                            caption = generate_caption(
                                features, 
                                caption_model, 
                                vocab, 
                                decoding_method, 
                                beam_width, 
                                max_length, 
                                device
                            )
                            
                            end_time = time.time()
                            st.session_state.processing_time = end_time - start_time
                        
                        if caption:
                            st.session_state.generated_caption = caption
                            st.rerun()
                        else:
                            st.error("Failed to generate caption. Please try again.")
                
                # Clear button
                if st.button("🗑️ Clear Image", use_container_width=True):
                    st.session_state.uploaded_image = None
                    st.session_state.generated_caption = ""
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            # Display example images
            st.markdown('<div class="info-box">No image uploaded yet. Please upload an image to generate captions.</div>', unsafe_allow_html=True)
            
            # Example captions
            st.markdown("### 📝 Example Captions")
            examples = [
                "A group of people riding bicycles on a sunny day in the city",
                "A golden retriever playing with a frisbee in a green park",
                "A beautiful sunset over snow-capped mountains with clouds",
                "Two children laughing while eating ice cream at the beach",
                "A chef preparing food in a modern kitchen with stainless steel appliances"
            ]
            
            for example in examples:
                st.markdown(f'<div class="example-caption">"{example}"</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📝 Generated Caption")
        
        if st.session_state.generated_caption:
            # Display caption with styling
            st.markdown(f'''
            <div class="caption-box">
                <h4>🎨 Generated Description:</h4>
                <p style="font-size: 1.2rem; line-height: 1.6;">"{st.session_state.generated_caption}"</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Performance metrics
            col_metrics1, col_metrics2 = st.columns(2)
            with col_metrics1:
                st.metric("Method", decoding_method)
            with col_metrics2:
                st.metric("Time", f"{st.session_state.processing_time:.2f}s")
            
            # Copy to clipboard
            caption_text = st.session_state.generated_caption
            col_copy, col_download = st.columns(2)
            
            with col_copy:
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.write(caption_text)
                    st.success("Caption copied to clipboard!")
            
            with col_download:
                st.download_button(
                    label="💾 Download Caption",
                    data=caption_text,
                    file_name="generated_caption.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Try another button
            if st.button("🔄 Generate Another", use_container_width=True):
                st.session_state.generated_caption = ""
                st.rerun()
            
            # Display uploaded image in this column too
            if st.session_state.uploaded_image:
                st.markdown("#### 📷 Your Image")
                st.image(st.session_state.uploaded_image, use_column_width=True)
        
        else:
            # Placeholder when no caption generated
            st.markdown('<div class="info-box">👈 Upload an image and click "Generate Caption" to see results here.</div>', unsafe_allow_html=True)
            
            # Quick info about methods
            st.markdown("### 🔍 About Methods")
            st.markdown("""
            **Greedy Search:**
            - Fastest method
            - Always picks most probable next word
            - Good for simple images
            
            **Beam Search:**
            - Explores multiple possibilities
            - Better for complex scenes
            - Adjust width for quality/speed trade-off
            """)
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("""
        **📚 Model Details**
        - Encoder: ResNet50
        - Decoder: LSTM
        - Hidden size: 512
        - Embedding: 256
        """)
    
    with col_footer2:
        st.markdown("""
        **⚡ Performance**
        - GPU accelerated
        - Real-time processing
        - Batch support ready
        - Memory efficient
        """)
    
    with col_footer3:
        st.markdown("""
        **🔧 Technical**
        - PyTorch backend
        - Streamlit frontend
        - REST API ready
        - Open source
        """)
    
    st.markdown("""
    <div style='text-align: center; color: #6B7280; margin-top: 2rem; padding: 1rem; border-top: 1px solid #E5E7EB;'>
        <p>Built with ❤️ using PyTorch & Streamlit | Neural Storyteller v1.0</p>
        <p style='font-size: 0.9rem;'>For best results, use high-quality images with clear subjects</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add debug info in expander
    with st.sidebar.expander("🔧 Debug Info"):
        st.write(f"Device: {device}")
        st.write(f"Model loaded: {caption_model is not None}")
        st.write(f"Vocabulary size: {len(vocab) if vocab else 0}")
        st.write(f"Session state keys: {list(st.session_state.keys())}")

if __name__ == "__main__":
    # Check if required files exist
    required_files = ["models/vocab.pkl", "models/best_model.pth"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"Missing required files: {', '.join(missing_files)}")
        st.info("""
        Please ensure you have:
        1. Trained model saved as `models/best_model.pth`
        2. Vocabulary saved as `models/vocab.pkl`
        
        You can train the model using the Jupyter notebook or download pre-trained weights.
        """)
        st.stop()
    
    main()