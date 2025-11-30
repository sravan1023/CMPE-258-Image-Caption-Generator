"""
Evaluation script for Image Captioning Models
Tests both Baseline and Attention models on BLEU, CIDEr, and METEOR metrics
"""

import torch
from torch.utils.data import DataLoader
import pickle
import os
import json
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
from torchvision import transforms

# Import models
from model_baseline_only import (
    BaselineCaptionModel,
    AttentionCaptionModel,
    get_data_loaders as get_baseline_loaders,
    get_data_loaders_attention,
)

# Metrics
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk


def get_inference_transform(image_size=224):
    """Image transform that mirrors validation preprocessing."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def preprocess_single_image(image_path, device, image_size=224):
    """Load and preprocess a single image for inference."""
    transform = get_inference_transform(image_size)
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def decode_indices(indices, vocab_data):
    """Convert list of token ids back to a caption string."""
    idx2word = vocab_data['idx2word']
    word2idx = vocab_data['word2idx']
    end_token = word2idx['<END>']
    start_token = word2idx['<START>']
    pad_token = word2idx['<PAD>']

    words = []
    for idx in indices:
        if idx == end_token:
            break
        if idx in (start_token, pad_token):
            continue
        word = idx2word.get(idx) or idx2word.get(str(idx), '<UNK>')
        words.append(word)
    return ' '.join(words)


def generate_caption(image_path, model, vocab_data, device, max_length=20, method='greedy', beam_width=3):
    """Generate a caption for an image path using the baseline model."""
    model.eval()
    with torch.no_grad():
        image_tensor = preprocess_single_image(image_path, device)
        start_token = vocab_data['word2idx']['<START>']
        end_token = vocab_data['word2idx']['<END>']
        preds = model.generate(
            image_tensor,
            max_length=max_length,
            start_token=start_token,
            end_token=end_token,
            method='beam' if method == 'beam' else 'greedy',
            beam_width=beam_width,
        )
        caption_tokens = preds[0].cpu().tolist()
    return decode_indices(caption_tokens, vocab_data)


def load_baseline_model(checkpoint_path, device):
    """Load baseline model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    vocab_data = checkpoint['vocab_data']
    
    model = BaselineCaptionModel(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=vocab_data['vocab_size'],
        dropout=config.get('dropout', 0.5),
        train_cnn=config.get('train_cnn', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded baseline model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, vocab_data


def load_attention_model(checkpoint_path, device):
    """Load attention model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    vocab_data = checkpoint['vocab_data']
    
    model = AttentionCaptionModel(
        embed_size=config['embed_size'],
        attention_dim=config['attention_dim'],
        decoder_dim=config['decoder_dim'],
        vocab_size=vocab_data['vocab_size'],
        encoder_dim=config['encoder_dim'],
        dropout=config['dropout'],
        train_cnn=config['train_cnn']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded attention model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, vocab_data


def generate_captions_baseline(model, data_loader, vocab_data, device, max_length=20):
    """Generate captions using baseline model"""
    model.eval()
    
    start_token = vocab_data['word2idx']['<START>']
    end_token = vocab_data['word2idx']['<END>']
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for images, captions, _ in tqdm(data_loader, desc="Generating captions (baseline)"):
            images = images.to(device)
            generated = model.generate(
                images,
                max_length=max_length,
                start_token=start_token,
                end_token=end_token,
                method='beam',
                beam_width=3
            )
            
            for i in range(images.size(0)):
                pred_indices = generated[i].cpu().tolist()
                pred_words = decode_indices(pred_indices, vocab_data).split()
                
                ref_indices = captions[i].cpu().tolist()
                ref_words = decode_indices(ref_indices, vocab_data).split()
                
                all_predictions.append(pred_words)
                all_references.append([ref_words])  # List of lists for multiple references
    
    return all_predictions, all_references


def generate_captions_attention(model, data_loader, vocab_data, device, max_length=20):
    """Generate captions using attention model"""
    model.eval()
    
    start_token = vocab_data['word2idx']['<START>']
    end_token = vocab_data['word2idx']['<END>']
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for images, captions, lengths in tqdm(data_loader, desc="Generating captions (attention)"):
            images = images.to(device)
            batch_size = images.size(0)
            
            encoder_out = model.encoder(images)
            generated, _ = model.decoder.sample(encoder_out, max_length, start_token, end_token)
            
            for i in range(batch_size):
                pred_indices = generated[i].cpu().tolist()
                pred_words = decode_indices(pred_indices, vocab_data).split()
                
                ref_indices = captions[i].cpu().tolist()
                ref_words = decode_indices(ref_indices, vocab_data).split()
                
                all_predictions.append(pred_words)
                all_references.append([ref_words])
    
    return all_predictions, all_references


def compute_bleu_scores(predictions, references):
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores"""
    smoothing = SmoothingFunction().method1
    
    bleu1 = corpus_bleu(references, predictions, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, predictions, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, predictions, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }


def compute_meteor_score(predictions, references):
    """Compute METEOR score"""
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        print("Downloading NLTK wordnet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    meteor_scores = []
    for pred, refs in zip(predictions, references):
        # METEOR expects tokenized inputs (lists of words)
        if refs:
            ref_tokens = refs[0]
        else:
            ref_tokens = []
        score = meteor_score([ref_tokens], pred)
        meteor_scores.append(score)
    
    return sum(meteor_scores) / len(meteor_scores)

try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False


def compute_cider_score(predictions, references):
    """Compute CIDEr score"""
    if not CIDER_AVAILABLE:
        return None
    
    # Import here to avoid unbound error
    from pycocoevalcap.cider.cider import Cider
    
    # Convert to format expected by CIDEr
    gts = {}  # ground truth captions
    res = {}  # predicted captions
    
    for i, (pred, refs) in enumerate(zip(predictions, references)):
        res[i] = [' '.join(pred)]
        gts[i] = [' '.join(ref) for ref in refs]
    
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    
    return score


def evaluate_model(model, data_loader, vocab_data, device, model_name, max_length=20):
    """Evaluate a model on all metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Generate captions
    if 'baseline' in model_name.lower():
        predictions, references = generate_captions_baseline(model, data_loader, vocab_data, device, max_length)
    else:
        predictions, references = generate_captions_attention(model, data_loader, vocab_data, device, max_length)
    
    print(f"\nGenerated {len(predictions)} captions")
    
    # Compute BLEU scores
    print("\nComputing BLEU scores...")
    bleu_scores = compute_bleu_scores(predictions, references)
    
    # Compute METEOR score
    print("Computing METEOR score...")
    meteor = compute_meteor_score(predictions, references)
    
    # Compute CIDEr score
    cider = None
    if CIDER_AVAILABLE:
        print("Computing CIDEr score...")
        cider = compute_cider_score(predictions, references)
    
    # Print results
    print(f"\n{model_name} Results:")
    print("-" * 40)
    for metric, score in bleu_scores.items():
        print(f"{metric:12s}: {score:.4f}")
    print(f"{'METEOR':12s}: {meteor:.4f}")
    if cider is not None:
        print(f"{'CIDEr':12s}: {cider:.4f}")
    
    results = {
        **bleu_scores,
        'METEOR': meteor
    }
    if cider is not None:
        results['CIDEr'] = cider
    
    # Show sample predictions
    print(f"\n{model_name} - Sample Predictions:")
    print("-" * 40)
    for i in range(min(5, len(predictions))):
        print(f"\nSample {i+1}:")
        print(f"Predicted: {' '.join(predictions[i])}")
        print(f"Reference: {' '.join(references[i][0])}")
    
    return results


def main():
    # Configuration
    baseline_checkpoint = './checkpoints/checkpoint_best.pth'
    attention_checkpoint = './checkpoints/checkpoint_attention_best.pth'
    image_dir = './raw_data/Images'
    processed_dir = './data'
    batch_size = 32
    max_length = 20
    save_results = './evaluation_results.json'
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Results dictionary
    all_results = {}
    
    # Evaluate baseline model
    if os.path.exists(baseline_checkpoint):
        print("\n" + "="*60)
        print("BASELINE MODEL EVALUATION")
        print("="*60)
        
        baseline_model, baseline_vocab = load_baseline_model(baseline_checkpoint, device)
        _, _, test_loader, _ = get_baseline_loaders(image_dir, processed_dir, batch_size, num_workers=0)
        
        baseline_results = evaluate_model(
            baseline_model, test_loader, baseline_vocab, 
            device, "Baseline Model", max_length
        )
        all_results['baseline'] = baseline_results
    else:
        print(f"Baseline checkpoint not found: {baseline_checkpoint}")
    
    # Evaluate attention model
    if os.path.exists(attention_checkpoint):
        print("\n" + "="*60)
        print("ATTENTION MODEL EVALUATION")
        print("="*60)
        
        attention_model, attention_vocab = load_attention_model(attention_checkpoint, device)
        _, _, test_loader, _ = get_data_loaders_attention(image_dir, processed_dir, batch_size, num_workers=0)
        
        attention_results = evaluate_model(
            attention_model, test_loader, attention_vocab,
            device, "Attention Model (Primary)", max_length
        )
        all_results['attention'] = attention_results
    else:
        print(f"Attention checkpoint not found: {attention_checkpoint}")
    
    # Save results
    if all_results:
        with open(save_results, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\n{'='*60}")
        print(f"Results saved to: {save_results}")
        print(f"{'='*60}")
        
        # Print comparison
        if len(all_results) == 2:
            print("\n" + "="*60)
            print("MODEL COMPARISON")
            print("="*60)
            print(f"{'Metric':<15} {'Baseline':<12} {'Attention':<12} {'Winner':<12}")
            print("-" * 60)
            
            for metric in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'CIDEr']:
                if metric in all_results['baseline'] and metric in all_results['attention']:
                    b_score = all_results['baseline'][metric]
                    a_score = all_results['attention'][metric]
                    winner = 'Attention' if a_score > b_score else 'Baseline' if b_score > a_score else 'Tie'
                    print(f"{metric:<15} {b_score:<12.4f} {a_score:<12.4f} {winner:<12}")


if __name__ == "__main__":
    main()

