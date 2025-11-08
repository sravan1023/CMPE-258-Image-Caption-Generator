"""
Inference and Evaluation script for Image Captioning Model
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class CaptionGenerator:
    """Generate captions for images"""
    
    def __init__(self, model, vocab_data, device='cuda'):
        """
        Args:
            model: Trained ImageCaptioningModel
            vocab_data: Vocabulary data dictionary
            device: Device to run inference on
        """
        self.model = model
        self.vocab_data = vocab_data
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.start_token = self.word2idx['<START>']
        self.end_token = self.word2idx['<END>']
        self.pad_token = self.word2idx['<PAD>']
        
        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path):
        """Load and preprocess an image"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # type: ignore
        return image_tensor.to(self.device)
    
    def generate_caption(self, image_path, method='greedy', beam_width=3, max_length=20):
        """
        Generate caption for an image
        Args:
            image_path: Path to image file
            method: 'greedy' or 'beam_search'
            beam_width: Beam width for beam search
            max_length: Maximum caption length
        Returns:
            caption: Generated caption as string
            caption_indices: Caption as list of indices
        """
        # Load image
        image = self.load_image(image_path)
        
        # Generate caption
        with torch.no_grad():
            caption_indices = self.model.generate_caption(
                image, 
                max_length=max_length,
                method=method,
                beam_width=beam_width,
                start_token=self.start_token,
                end_token=self.end_token
            )
        
        # Convert indices to words
        caption_indices = caption_indices.cpu().numpy().tolist()
        caption = self.indices_to_caption(caption_indices)
        
        return caption, caption_indices
    
    def indices_to_caption(self, indices):
        """Convert indices to caption string"""
        words = []
        for idx in indices:
            if idx == self.end_token:
                break
            if idx not in [self.pad_token, self.start_token]:
                word = self.idx2word.get(str(idx), '<UNK>')
                words.append(word)
        return ' '.join(words)
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Generate and visualize caption for an image
        Args:
            image_path: Path to image
            save_path: Path to save visualization (optional)
        """
        # Generate captions using both methods
        caption_greedy, _ = self.generate_caption(image_path, method='greedy')
        caption_beam, _ = self.generate_caption(image_path, method='beam_search', beam_width=3)
        
        # Load and display image
        image = Image.open(image_path)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(np.array(image))  # Convert PIL Image to numpy array
        plt.axis('off')
        plt.title(f'Greedy: {caption_greedy}\n\nBeam Search: {caption_beam}', 
                 fontsize=12, wrap=True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return caption_greedy, caption_beam


class CaptionEvaluator:
    """Evaluate caption generation with BLEU scores"""
    
    def __init__(self, model, data_loader, vocab_data, device='cuda'):
        """
        Args:
            model: Trained model
            data_loader: Data loader for evaluation
            vocab_data: Vocabulary data
            device: Device to run evaluation on
        """
        self.model = model
        self.data_loader = data_loader
        self.vocab_data = vocab_data
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.idx2word = vocab_data['idx2word']
        self.start_token = vocab_data['word2idx']['<START>']
        self.end_token = vocab_data['word2idx']['<END>']
        self.pad_token = vocab_data['word2idx']['<PAD>']
    
    def indices_to_words(self, indices):
        """Convert indices to list of words"""
        words = []
        for idx in indices:
            if idx == self.end_token:
                break
            if idx not in [self.pad_token, self.start_token]:
                word = self.idx2word.get(str(idx), '<UNK>')
                words.append(word)
        return words
    
    def calculate_bleu(self, reference, candidate, n=4):
        """
        Calculate BLEU score
        Args:
            reference: List of reference word lists
            candidate: List of candidate words
            n: Maximum n-gram order
        Returns:
            BLEU score
        """
        from collections import Counter
        
        def get_ngrams(words, n):
            """Get n-grams from word list"""
            return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        
        # Calculate precision for each n-gram order
        precisions = []
        
        for i in range(1, n+1):
            candidate_ngrams = Counter(get_ngrams(candidate, i))
            
            # Maximum count for each n-gram across all references
            max_ref_counts = Counter()
            for ref in reference:
                ref_ngrams = Counter(get_ngrams(ref, i))
                for ngram in ref_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
            
            # Clipped counts
            clipped_counts = {
                ngram: min(count, max_ref_counts[ngram])
                for ngram, count in candidate_ngrams.items()
            }
            
            # Precision
            numerator = sum(clipped_counts.values())
            denominator = max(1, sum(candidate_ngrams.values()))
            precision = numerator / denominator if denominator > 0 else 0
            precisions.append(precision)
        
        # Brevity penalty
        ref_lengths = [len(ref) for ref in reference]
        closest_ref_len = min(ref_lengths, key=lambda x: abs(x - len(candidate)))
        
        if len(candidate) > closest_ref_len:
            bp = 1
        else:
            bp = np.exp(1 - closest_ref_len / max(1, len(candidate)))
        
        # Geometric mean of precisions
        if min(precisions) > 0:
            log_precisions = [np.log(p) for p in precisions]
            geo_mean = np.exp(sum(log_precisions) / len(log_precisions))
            bleu = bp * geo_mean
        else:
            bleu = 0
        
        return bleu, precisions
    
    def evaluate(self, num_samples=None):
        """
        Evaluate model on dataset
        Args:
            num_samples: Number of samples to evaluate (None for all)
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model...")
        
        bleu_scores = {1: [], 2: [], 3: [], 4: []}
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, captions, lengths) in enumerate(tqdm(self.data_loader)):
                if num_samples and total_samples >= num_samples:
                    break
                
                # Move to device
                images = images.to(self.device)
                
                # Generate captions
                for i in range(images.size(0)):
                    if num_samples and total_samples >= num_samples:
                        break
                    
                    image = images[i:i+1]
                    reference_indices = captions[i].cpu().numpy().tolist()
                    
                    # Generate caption
                    predicted_indices = self.model.generate_caption(
                        image,
                        max_length=self.vocab_data['max_caption_length'],
                        method='greedy',
                        start_token=self.start_token,
                        end_token=self.end_token
                    )
                    
                    # Convert to words
                    reference_words = self.indices_to_words(reference_indices)
                    predicted_words = self.indices_to_words(predicted_indices.cpu().numpy().tolist())
                    
                    # Calculate BLEU scores
                    bleu, precisions = self.calculate_bleu([reference_words], predicted_words, n=4)
                    
                    for n in range(1, 5):
                        bleu_n, _ = self.calculate_bleu([reference_words], predicted_words, n=n)
                        bleu_scores[n].append(bleu_n)
                    
                    total_samples += 1
        
        # Calculate average scores
        results = {
            f'BLEU-{n}': np.mean(scores) for n, scores in bleu_scores.items()
        }
        results['num_samples'] = float(total_samples)  # type: ignore
        
        print("\nEvaluation Results:")
        print("-" * 40)
        for metric, score in results.items():
            if metric != 'num_samples':
                print(f"{metric}: {score:.4f}")
        print(f"Evaluated on {results['num_samples']} samples")
        
        return results


def load_model_from_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    from model import ImageCaptioningModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    vocab_data = checkpoint['vocab_data']
    
    # Create model
    model = ImageCaptioningModel(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=vocab_data['vocab_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        train_cnn=config['train_cnn']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, vocab_data, config


if __name__ == "__main__":
    import sys
    
    # Paths
    checkpoint_path = "./checkpoints/checkpoint_best.pth"
    image_dir = "../raw_data/Images"
    processed_data_dir = "../data"
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, vocab_data, config = load_model_from_checkpoint(checkpoint_path, device)
    
    # Create caption generator
    generator = CaptionGenerator(model, vocab_data, device)
    
    # Example: Generate caption for a single image
    test_image = os.path.join(image_dir, "1000268201_693b08cb0e.jpg")
    if os.path.exists(test_image):
        print(f"\nGenerating caption for: {test_image}")
        caption_greedy, caption_beam = generator.visualize_prediction(
            test_image,
            save_path="./sample_prediction.png"
        )
        print(f"Greedy: {caption_greedy}")
        print(f"Beam Search: {caption_beam}")
    
    # Evaluate on test set
    from dataset import get_data_loaders
    
    _, _, test_loader, _ = get_data_loaders(
        image_dir=image_dir,
        processed_data_dir=processed_data_dir,
        batch_size=32,
        num_workers=2,
        image_size=224
    )
    
    evaluator = CaptionEvaluator(model, test_loader, vocab_data, device)
    results = evaluator.evaluate(num_samples=100)  # Evaluate on 100 samples
