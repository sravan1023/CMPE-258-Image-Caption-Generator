"""
CNN-LSTM Encoder-Decoder Model for Image Captioning
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """CNN Encoder using pretrained ResNet"""
    
    def __init__(self, embed_size, train_cnn=False):
        """
        Args:
            embed_size: Dimensionality of image embeddings
            train_cnn: Whether to fine-tune the CNN
        """
        super(EncoderCNN, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Add a linear layer to transform ResNet output to embed_size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        
        # Freeze CNN parameters if not training
        if not train_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: Image tensor of shape (batch_size, 3, 224, 224)
        Returns:
            features: Image features of shape (batch_size, embed_size)
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        
        return features


class DecoderLSTM(nn.Module):
    """LSTM Decoder for generating captions"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        """
        Args:
            embed_size: Dimensionality of image and word embeddings
            hidden_size: Number of features in hidden state of LSTM
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DecoderLSTM, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to project LSTM outputs to vocabulary size
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """
        Args:
            features: Image features from encoder (batch_size, embed_size)
            captions: Caption tokens (batch_size, caption_length)
        Returns:
            outputs: Predicted scores for each word in vocabulary (batch_size, caption_length, vocab_size)
        """
        # Embed captions (exclude the last token for input)
        embeddings = self.embed(captions[:, :-1])
        
        # Concatenate image features with caption embeddings
        # Image features serve as the first "word"
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings), dim=1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embeddings)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary size
        outputs = self.fc(lstm_out)
        
        return outputs
    
    def sample(self, features, max_length=20, start_token=1, end_token=2):
        """
        Generate captions using greedy search
        Args:
            features: Image features (batch_size, embed_size)
            max_length: Maximum caption length
            start_token: Index of <START> token
            end_token: Index of <END> token
        Returns:
            captions: Generated caption indices (batch_size, max_length)
        """
        batch_size = features.size(0)
        captions = []
        
        # Initialize LSTM states
        states = None
        inputs = features.unsqueeze(1)
        
        for _ in range(max_length):
            # Forward pass through LSTM
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_out.squeeze(1))
            
            # Get predicted word
            predicted = outputs.argmax(dim=1)
            captions.append(predicted)
            
            # Prepare input for next step
            inputs = self.embed(predicted).unsqueeze(1)
        
        # Stack captions
        captions = torch.stack(captions, dim=1)
        
        return captions
    
    def sample_beam_search(self, features, beam_width=3, max_length=20, 
                          start_token=1, end_token=2):
        """
        Generate captions using beam search
        Args:
            features: Image features (1, embed_size) - beam search for single image
            beam_width: Number of beams to keep
            max_length: Maximum caption length
            start_token: Index of <START> token
            end_token: Index of <END> token
        Returns:
            caption: Generated caption indices (max_length,)
        """
        # Initialize beam with start token
        k = beam_width
        
        # Initial input
        inputs = features.unsqueeze(1)
        states = None
        
        # First step
        lstm_out, states = self.lstm(inputs, states)
        outputs = self.fc(lstm_out.squeeze(1))
        log_probs = torch.log_softmax(outputs, dim=1)
        
        # Get top k tokens
        top_log_probs, top_indices = log_probs.topk(k, dim=1)
        
        # Initialize beams: (sequence, log_prob, states)
        beams = []
        for i in range(k):
            seq = [top_indices[0, i].item()]
            log_prob = top_log_probs[0, i].item()
            beams.append((seq, log_prob, states))
        
        completed_beams = []
        
        for _ in range(max_length - 1):
            new_beams = []
            
            for seq, log_prob, curr_states in beams:
                # If sequence ended, add to completed
                if seq[-1] == end_token:
                    completed_beams.append((seq, log_prob))
                    continue
                
                # Get last word and embed it
                last_word = torch.LongTensor([seq[-1]]).to(features.device)
                inputs = self.embed(last_word).unsqueeze(1)
                
                # LSTM step
                lstm_out, new_states = self.lstm(inputs, curr_states)
                outputs = self.fc(lstm_out.squeeze(1))
                next_log_probs = torch.log_softmax(outputs, dim=1)
                
                # Get top k next words
                top_next_log_probs, top_next_indices = next_log_probs.topk(k, dim=1)
                
                # Create new beam candidates
                for i in range(k):
                    new_seq = seq + [top_next_indices[0, i].item()]
                    new_log_prob = log_prob + top_next_log_probs[0, i].item()
                    new_beams.append((new_seq, new_log_prob, new_states))
            
            # If all beams completed, break
            if not new_beams:
                break
            
            # Keep top k beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:k]
            beams = new_beams
        
        # Add remaining beams to completed
        for seq, log_prob, _ in beams:
            completed_beams.append((seq, log_prob))
        
        # Return best beam
        if completed_beams:
            best_seq, _ = max(completed_beams, key=lambda x: x[1])
            return torch.LongTensor(best_seq)
        else:
            return torch.LongTensor(beams[0][0])


class ImageCaptioningModel(nn.Module):
    """Complete CNN-LSTM Image Captioning Model"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, 
                 dropout=0.5, train_cnn=False):
        """
        Args:
            embed_size: Dimensionality of image and word embeddings
            hidden_size: Number of features in hidden state of LSTM
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            train_cnn: Whether to fine-tune the CNN encoder
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.encoder = EncoderCNN(embed_size, train_cnn)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, 
                                   num_layers, dropout)
    
    def forward(self, images, captions):
        """
        Args:
            images: Image tensor (batch_size, 3, 224, 224)
            captions: Caption tokens (batch_size, caption_length)
        Returns:
            outputs: Predicted scores (batch_size, caption_length, vocab_size)
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def generate_caption(self, image, max_length=20, method='greedy', 
                        beam_width=3, start_token=1, end_token=2):
        """
        Generate caption for a single image
        Args:
            image: Image tensor (1, 3, 224, 224)
            max_length: Maximum caption length
            method: 'greedy' or 'beam_search'
            beam_width: Beam width for beam search
            start_token: Index of <START> token
            end_token: Index of <END> token
        Returns:
            caption: Generated caption indices
        """
        self.eval()
        with torch.no_grad():
            features = self.encoder(image)
            
            if method == 'greedy':
                caption = self.decoder.sample(features, max_length, 
                                             start_token, end_token)
                return caption[0]
            elif method == 'beam_search':
                caption = self.decoder.sample_beam_search(features, beam_width, 
                                                         max_length, start_token, end_token)
                return caption
            else:
                raise ValueError(f"Unknown sampling method: {method}")


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    embed_size = 256
    hidden_size = 512
    vocab_size = 5000
    max_caption_length = 20
    
    # Create model
    model = ImageCaptioningModel(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=1,
        dropout=0.5,
        train_cnn=False
    )
    
    # Test forward pass
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, max_caption_length))
    
    outputs = model(images, captions)
    print(f"Output shape: {outputs.shape}")
    print(f"Expected: ({batch_size}, {max_caption_length}, {vocab_size})")
    
    # Test caption generation
    single_image = torch.randn(1, 3, 224, 224)
    generated_caption = model.generate_caption(single_image, method='greedy')
    print(f"\nGenerated caption shape: {generated_caption.shape}")
