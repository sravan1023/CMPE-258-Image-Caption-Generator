"""
CNN-LSTM Encoder-Decoder Model with Attention Mechanism for Image Captioning
This implements Bahdanau-style attention for visual features
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNNAttention(nn.Module):
    """CNN Encoder that outputs spatial feature maps for attention"""
    
    def __init__(self, embed_size, train_cnn=False):
        """
        Args:
            embed_size: Dimensionality of image embeddings
            train_cnn: Whether to fine-tune the CNN
        """
        super(EncoderCNNAttention, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove average pooling and FC layer to keep spatial features
        # We want features from the last conv layer (before pooling)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        
        # Get number of features from conv layer (2048 for ResNet-50)
        self.resnet_feature_size = 2048
        
        # Project features to embed_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Output: (batch, 2048, 7, 7)
        self.projection = nn.Linear(self.resnet_feature_size, embed_size)
        
        # Freeze CNN parameters if not training
        if not train_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: Image tensor of shape (batch_size, 3, 224, 224)
        Returns:
            features: Spatial features of shape (batch_size, num_pixels, embed_size)
                     where num_pixels = 49 (7x7)
        """
        # Extract features from ResNet
        features = self.resnet(images)  # (batch, 2048, 7, 7)
        features = self.adaptive_pool(features)  # Ensure (batch, 2048, 7, 7)
        
        # Reshape to (batch, num_pixels, feature_dim)
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # (batch, 7, 7, 2048)
        features = features.view(batch_size, -1, self.resnet_feature_size)  # (batch, 49, 2048)
        
        # Project to embed_size
        features = self.projection(features)  # (batch, 49, embed_size)
        
        return features


class AttentionModule(nn.Module):
    """Bahdanau Attention mechanism"""
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Args:
            encoder_dim: Feature size of encoded images
            decoder_dim: Size of decoder's hidden state
            attention_dim: Size of attention network
        """
        super(AttentionModule, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out: Encoded images (batch_size, num_pixels, encoder_dim)
            decoder_hidden: Previous decoder hidden state (batch_size, decoder_dim)
        Returns:
            attention_weighted_encoding: Weighted encoding (batch_size, encoder_dim)
            alpha: Attention weights (batch_size, num_pixels)
        """
        # Project encoder output
        att1 = self.encoder_att(encoder_out)  # (batch, num_pixels, attention_dim)
        
        # Project decoder hidden state
        att2 = self.decoder_att(decoder_hidden)  # (batch, attention_dim)
        
        # Add and apply activation
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1)))  # (batch, num_pixels, 1)
        
        # Calculate attention weights
        alpha = self.softmax(att.squeeze(2))  # (batch, num_pixels)
        
        # Apply attention weights to encoder output
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch, encoder_dim)
        
        return attention_weighted_encoding, alpha


class DecoderLSTMAttention(nn.Module):
    """LSTM Decoder with Attention for generating captions"""
    
    def __init__(self, embed_size, attention_dim, decoder_dim, vocab_size, 
                 encoder_dim=256, dropout=0.5):
        """
        Args:
            embed_size: Dimensionality of word embeddings
            attention_dim: Size of attention network
            decoder_dim: Size of decoder's LSTM hidden state
            vocab_size: Size of vocabulary
            encoder_dim: Feature size of encoded images
            dropout: Dropout probability
        """
        super(DecoderLSTMAttention, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_size = embed_size
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        # Attention network
        self.attention = AttentionModule(encoder_dim, decoder_dim, attention_dim)
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        # LSTM decoder
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, decoder_dim)
        
        # Linear layers to find initial hidden state and cell state
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Linear layer to find scores over vocabulary
        self.fc = nn.Linear(decoder_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize embeddings and linear layers with small random values"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out):
        """
        Create initial hidden and cell states for decoder LSTM
        Args:
            encoder_out: Encoded images (batch_size, num_pixels, encoder_dim)
        Returns:
            h: Initial hidden state (batch_size, decoder_dim)
            c: Initial cell state (batch_size, decoder_dim)
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Args:
            encoder_out: Encoded images (batch_size, num_pixels, encoder_dim)
            encoded_captions: Encoded captions (batch_size, max_caption_length)
            caption_lengths: Caption lengths (batch_size, 1)
        Returns:
            predictions: Scores for vocabulary (batch_size, max_caption_length, vocab_size)
            alphas: Attention weights (batch_size, max_caption_length, num_pixels)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten image
        num_pixels = encoder_out.size(1)
        
        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embed captions
        embeddings = self.embedding(encoded_captions)  # (batch, max_caption_length, embed_size)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # We won't decode at the <end> position, since we've finished generating
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)
        
        # At each time-step, decode by attention-weighting encoder output
        for t in range(max(decode_lengths)):
            # Find batch size at this timestep
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Attention mechanism
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t]
            )
            
            # Gating scalar (sentinel)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM step
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            # Predict next word
            preds = self.fc(self.dropout_layer(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, alphas, encoded_captions, decode_lengths, sort_ind
    
    def sample(self, encoder_out, max_length=20, start_token=1, end_token=2):
        """
        Generate captions using greedy search
        Args:
            encoder_out: Encoded images (batch_size, num_pixels, encoder_dim)
            max_length: Maximum caption length
            start_token: Index of <START> token
            end_token: Index of <END> token
        Returns:
            captions: Generated caption indices (batch_size, max_length)
            alphas: Attention weights (batch_size, max_length, num_pixels)
        """
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # Start with <START> token
        inputs = torch.LongTensor([[start_token]] * batch_size).to(encoder_out.device)
        
        captions = []
        alphas = []
        
        for t in range(max_length):
            # Embed current input
            embeddings = self.embedding(inputs).squeeze(1)  # (batch, embed_size)
            
            # Attention
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            
            # Gating
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTM step
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )
            
            # Predict next word
            scores = self.fc(h)
            predicted = scores.argmax(dim=1)
            
            captions.append(predicted)
            alphas.append(alpha)
            
            # Prepare next input
            inputs = predicted.unsqueeze(1)
        
        captions = torch.stack(captions, dim=1)  # (batch, max_length)
        alphas = torch.stack(alphas, dim=1)  # (batch, max_length, num_pixels)
        
        return captions, alphas


class ImageCaptioningModelAttention(nn.Module):
    """Complete CNN-LSTM Image Captioning Model with Attention"""
    
    def __init__(self, embed_size, attention_dim, decoder_dim, vocab_size, 
                 encoder_dim=256, dropout=0.5, train_cnn=False):
        """
        Args:
            embed_size: Dimensionality of word embeddings
            attention_dim: Size of attention network
            decoder_dim: Size of decoder's LSTM
            vocab_size: Size of vocabulary
            encoder_dim: Feature size after projection
            dropout: Dropout probability
            train_cnn: Whether to fine-tune the CNN encoder
        """
        super(ImageCaptioningModelAttention, self).__init__()
        
        self.encoder = EncoderCNNAttention(encoder_dim, train_cnn)
        self.decoder = DecoderLSTMAttention(
            embed_size=embed_size,
            attention_dim=attention_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout
        )
    
    def forward(self, images, captions, caption_lengths):
        """
        Args:
            images: Image tensor (batch_size, 3, 224, 224)
            captions: Caption tokens (batch_size, max_caption_length)
            caption_lengths: Caption lengths (batch_size, 1)
        Returns:
            predictions: Predicted scores (batch_size, max_caption_length, vocab_size)
            alphas: Attention weights (batch_size, max_caption_length, num_pixels)
            encoded_captions: Sorted captions
            decode_lengths: Decode lengths
            sort_ind: Sort indices
        """
        encoder_out = self.encoder(images)
        predictions, alphas, encoded_captions, decode_lengths, sort_ind = self.decoder(
            encoder_out, captions, caption_lengths
        )
        return predictions, alphas, encoded_captions, decode_lengths, sort_ind
    
    def generate_caption(self, image, max_length=20, start_token=1, end_token=2):
        """
        Generate caption for a single image
        Args:
            image: Image tensor (1, 3, 224, 224)
            max_length: Maximum caption length
            start_token: Index of <START> token
            end_token: Index of <END> token
        Returns:
            caption: Generated caption indices
            alphas: Attention weights
        """
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(image)
            caption, alphas = self.decoder.sample(
                encoder_out, max_length, start_token, end_token
            )
            return caption[0], alphas[0]


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    embed_size = 256
    attention_dim = 512
    decoder_dim = 512
    vocab_size = 5000
    max_caption_length = 20
    
    # Create model
    model = ImageCaptioningModelAttention(
        embed_size=embed_size,
        attention_dim=attention_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=256,
        dropout=0.5,
        train_cnn=False
    )
    
    # Test forward pass
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, max_caption_length))
    caption_lengths = torch.LongTensor([max_caption_length] * batch_size).unsqueeze(1)
    
    predictions, alphas, _, _, _ = model(images, captions, caption_lengths)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Alphas shape: {alphas.shape}")
    print(f"Expected predictions: ({batch_size}, {max_caption_length-1}, {vocab_size})")
    print(f"Expected alphas: ({batch_size}, {max_caption_length-1}, 49)")
    
    # Test caption generation
    single_image = torch.randn(1, 3, 224, 224)
    generated_caption, attention_weights = model.generate_caption(single_image)
    print(f"\nGenerated caption shape: {generated_caption.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
