import torch.nn as nn
import torch

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