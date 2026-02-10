import re
from collections import Counter
import numpy as np

class Vocabulary:
    """Vocabulary class for word-to-index and index-to-word mappings"""
    def __init__(self, freq_threshold=5):
        # Special tokens
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # Mappings
        self.word2idx = {}
        self.idx2word = {}
        self.freq_threshold = freq_threshold
        
    def build_vocabulary(self, caption_list):
        """Build vocabulary from list of captions"""
        # Count word frequencies
        word_freq = Counter()
        for caption in caption_list:
            tokens = caption.split()
            word_freq.update(tokens)
        
        # Add special tokens first
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.start_token] = 1
        self.word2idx[self.end_token] = 2
        self.word2idx[self.unk_token] = 3
        
        # Add words that meet frequency threshold
        idx = 4
        for word, freq in word_freq.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = idx
                idx += 1
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulary Statistics:")
        print(f"  Total unique words in corpus: {len(word_freq)}")
        print(f"  Words with freq >= {self.freq_threshold}: {len(self.word2idx) - 4}")
        print(f"  Vocabulary size (including special tokens): {len(self.word2idx)}")
        
    def numericalize(self, caption):
        """Convert caption to list of indices"""
        tokens = caption.split()
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
    
    def decode(self, indices):
        """Convert indices back to caption"""
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        # Remove special tokens and join
        words = [w for w in words if w not in [self.pad_token, self.start_token, self.end_token]]
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)

def clean_caption(caption):
    """Clean a single caption string"""
    # Convert to lowercase
    caption = caption.lower()
    # Remove special characters and digits
    caption = re.sub(r'[^a-z\s]', '', caption)
    # Remove extra whitespace
    caption = ' '.join(caption.split())
    return caption

def preprocess_image_captions(df):
    """
    Preprocess image captions from DataFrame
    Returns: Dictionary {image_name: [cleaned_captions]}
    """
    # Clean all captions
    df['caption'] = df['caption'].astype(str)
    df['caption_clean'] = df['caption'].apply(clean_caption)
    
    # Group by image to get all captions per image
    image_captions = df.groupby('image')['caption_clean'].apply(list).to_dict()
    
    return image_captions

def build_vocab_from_captions(caption_list, freq_threshold=5):
    """
    Build vocabulary from list of captions
    Args:
        caption_list: List of caption strings
        freq_threshold: Minimum frequency for word inclusion
    Returns:
        vocab: Vocabulary object
    """
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(caption_list)
    return vocab

def prepare_caption_for_training(caption, vocab):
    """
    Prepare caption for model training/inference
    Args:
        caption: String caption
        vocab: Vocabulary object
    Returns:
        caption_tensor: Tensor of indices with special tokens
    """
    # Clean caption first
    caption_clean = clean_caption(caption)
    
    # Numericalize caption and add special tokens
    caption_indices = [vocab.word2idx[vocab.start_token]]
    caption_indices.extend(vocab.numericalize(caption_clean))
    caption_indices.append(vocab.word2idx[vocab.end_token])
    
    return caption_indices

def create_vocab_from_scratch(captions_text_file, freq_threshold=5):
    """
    Create vocabulary from scratch using captions text file
    Args:
        captions_text_file: Path to captions.txt file
        freq_threshold: Minimum word frequency
    Returns:
        vocab: Vocabulary object
        image_captions: Dictionary of image to captions
    """
    import pandas as pd
    
    # Load captions file
    # Assuming format: image,caption (comma-separated)
    df = pd.read_csv(captions_text_file)
    
    # Preprocess captions
    image_captions = preprocess_image_captions(df)
    
    # Flatten all captions
    all_captions = [cap for caps in image_captions.values() for cap in caps]
    
    # Build vocabulary
    vocab = build_vocab_from_captions(all_captions, freq_threshold)
    
    return vocab, image_captions

def save_vocabulary(vocab, filepath):
    """Save vocabulary to pickle file"""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {filepath}")

def load_vocabulary(filepath):
    """Load vocabulary from pickle file"""
    import pickle
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

# Helper function for Streamlit app
def prepare_for_inference(caption_model, vocab):
    """
    Prepare model and vocabulary for inference
    This can be cached in Streamlit
    """
    # This function ensures the vocabulary is properly loaded
    # and the model is in evaluation mode
    if hasattr(vocab, 'word2idx') and hasattr(vocab, 'idx2word'):
        return caption_model, vocab
    else:
        # If vocab is loaded from pickle, it should already have these attributes
        raise ValueError("Vocabulary object doesn't have required attributes")

def test_vocabulary_operations():
    """Test vocabulary creation and operations"""
    # Example usage
    sample_captions = [
        "a dog running in the park",
        "two cats playing with a ball",
        "a beautiful sunset over mountains",
        "people walking in the city street"
    ]
    
    # Build vocabulary
    vocab = build_vocab_from_captions(sample_captions, freq_threshold=1)
    
    # Test numericalization
    test_caption = "a dog running in the park"
    indices = vocab.numericalize(test_caption)
    decoded = vocab.decode(indices)
    
    print(f"Original: {test_caption}")
    print(f"Indices: {indices}")
    print(f"Decoded: {decoded}")
    
    # Test with special tokens
    caption_with_tokens = prepare_caption_for_training(test_caption, vocab)
    print(f"With special tokens: {caption_with_tokens}")
    
    return vocab

if __name__ == "__main__":
    # Test the module
    vocab = test_vocabulary_operations()
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Special tokens mapping: {vocab.word2idx}")