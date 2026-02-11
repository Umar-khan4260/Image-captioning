"""
Neural Storyteller - Image Captioning Streamlit App
Deploy your trained image captioning model
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Neural Storyteller - Image Captioning",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
    }
    .caption-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MODEL CLASSES (Same as in training)
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, image_feature_dim=2048, hidden_size=512, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(image_feature_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, image_features):
        hidden = self.fc(image_features)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=1, dropout=0.5):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, captions, encoder_hidden):
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)
        
        h0 = encoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.fc(lstm_out)
        
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, num_layers=1, dropout=0.5):
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = Encoder(
            image_feature_dim=2048,
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, image_features, captions):
        hidden_state = self.encoder(image_features)
        outputs = self.decoder(captions, hidden_state)
        return outputs

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def beam_search(model, image_features, vocab, beam_width=3, max_length=50, device='cpu'):
    """Generate caption using beam search"""
    model.eval()
    
    with torch.no_grad():
        image_features = image_features.to(device)
        hidden_state = model.encoder(image_features)
        
        start_token = vocab.word2idx[vocab.start_token]
        end_token = vocab.word2idx[vocab.end_token]
        
        h = hidden_state.unsqueeze(0)
        c = torch.zeros_like(h)
        
        beams = [([start_token], 0.0, h, c)]
        completed_beams = []
        
        for step in range(max_length):
            all_candidates = []
            
            for sequence, score, h_state, c_state in beams:
                if sequence[-1] == end_token:
                    completed_beams.append((sequence, score))
                    continue
                
                last_word = torch.tensor([[sequence[-1]]]).to(device)
                embeddings = model.decoder.embedding(last_word)
                lstm_out, (h_new, c_new) = model.decoder.lstm(embeddings, (h_state, c_state))
                output = model.decoder.fc(lstm_out.squeeze(1))
                
                log_probs = torch.log_softmax(output, dim=1)
                top_probs, top_indices = log_probs.topk(beam_width)
                
                for i in range(beam_width):
                    word_idx = top_indices[0][i].item()
                    word_prob = top_probs[0][i].item()
                    new_sequence = sequence + [word_idx]
                    new_score = score + word_prob
                    all_candidates.append((new_sequence, new_score, h_new, c_new))
            
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if all(seq[-1] == end_token for seq, _, _, _ in beams):
                completed_beams.extend([(seq, score) for seq, score, _, _ in beams])
                break
        
        completed_beams.extend([(seq, score) for seq, score, _, _ in beams])
        
        if completed_beams:
            best_sequence = max(completed_beams, key=lambda x: x[1])[0]
        else:
            best_sequence = beams[0][0]
        
        caption_indices = [idx for idx in best_sequence if idx not in [start_token, end_token]]
        caption = vocab.decode(caption_indices)
        return caption

# ============================================================================
# LOAD MODEL AND RESOURCES
# ============================================================================

@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_file = 'complete_model_package.pkl'
    
    # Download model if not present
    if not os.path.exists(model_file):
        st.info("📥 Downloading model (one-time setup, ~2 minutes)...")
        try:
            import gdown
            # TODO: Replace with your Google Drive file ID
            file_id = '1LJPAF_zo6AN_xnX8uCgidNRLEbuaQa4B'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_file, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            st.info("Please upload the model file manually or check the file ID.")
            return None, None, device
    
    # Load model package
    try:
        with open(model_file, 'rb') as f:
            model_package = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model file: {str(e)}")
        st.info("The model file may be corrupted. Please re-download from Kaggle.")
        return None, None, device
    
    vocab = model_package['vocab']
    config = model_package['config']
    
    # Initialize model
    model = ImageCaptioningModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(model_package['model_state_dict'])
    model.eval()
    
    return model, vocab, device

@st.cache_resource
def load_feature_extractor():
    """Load ResNet50 for feature extraction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()
    
    return resnet, device

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image):
    """Preprocess uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and description
    st.title("🖼️ Neural Storyteller")
    st.markdown("### AI-Powered Image Captioning")
    st.markdown("Upload an image and let AI generate a natural language description!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        search_method = st.radio(
            "Search Method",
            ["Beam Search", "Greedy Search"],
            help="Beam Search usually produces better captions"
        )
        
        if search_method == "Beam Search":
            beam_width = st.slider(
                "Beam Width",
                min_value=1,
                max_value=5,
                value=3,
                help="Higher values may produce better captions but are slower"
            )
        else:
            beam_width = 1
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses a **Seq2Seq model** with:
        - **Encoder**: ResNet50 CNN
        - **Decoder**: LSTM RNN
        - **Dataset**: Flickr30k
        - **Vocabulary**: ~8,000 words
        """)
        
        st.markdown("---")
        st.markdown("### Model Performance")
        st.markdown("""
        - BLEU-4: Check evaluation
        - METEOR: Check evaluation
        - Trained for 10 epochs
        """)
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a JPG, JPEG, or PNG image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary"):
                with st.spinner('Generating caption...'):
                    try:
                        # Load model and resources
                        model, vocab, model_device = load_model_and_vocab()
                        
                        # Check if model loaded successfully
                        if model is None or vocab is None:
                            st.error("❌ Model not loaded. Please check the error messages above.")
                            st.stop()
                        
                        resnet, resnet_device = load_feature_extractor()
                        
                        # Preprocess image
                        img_tensor = preprocess_image(image).to(resnet_device)
                        
                        # Extract features
                        with torch.no_grad():
                            features = resnet(img_tensor).view(1, -1)
                        
                        # Generate caption
                        caption = beam_search(
                            model, 
                            features, 
                            vocab, 
                            beam_width=beam_width, 
                            device=model_device
                        )
                        
                        # Display caption
                        st.markdown("### 📝 Generated Caption:")
                        st.markdown(f'<div class="caption-box"><h3>{caption.capitalize()}</h3></div>', 
                                  unsafe_allow_html=True)
                        
                        # Success message
                        st.success("✅ Caption generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        st.error("Please make sure all model files are present.")
                        import traceback
                        st.code(traceback.format_exc())
    
    else:
        # Placeholder when no image is uploaded
        st.info("👆 Upload an image to get started!")
        
        # Example images section
        st.markdown("---")
        st.markdown("### 📸 Example Use Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌅 Landscapes**")
            st.markdown("Mountains, beaches, forests")
        
        with col2:
            st.markdown("**🐕 Animals**")
            st.markdown("Pets, wildlife, birds")
        
        with col3:
            st.markdown("**👥 People**")
            st.markdown("Activities, portraits, groups")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using PyTorch, Streamlit, and Flickr30k dataset</p>
        <p>Model: Encoder-Decoder with LSTM | Architecture: ResNet50 + LSTM</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
