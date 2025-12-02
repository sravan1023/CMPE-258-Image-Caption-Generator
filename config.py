"""
Configuration file for hyperparameter tuning
"""

# Data paths
DATA_DIR = './data'
IMAGE_DIR = './raw_data/Images'
CAPTIONS_FILE = './raw_data/captions.txt'
CHECKPOINT_DIR = './checkpoints'

# Preprocessing
MIN_WORD_FREQ = 5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# Baseline model hyperparameters
BASELINE = {
    'embed_size': 256,
    'hidden_size': 512,
    'dropout': 0.5,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 20,
    'weight_decay': 1e-5,
    'grad_clip': 5.0
}

# primary model hyperparameters
ATTENTION = {
    'embed_size': 512,       
    'attention_dim': 512,
    'decoder_dim': 512,
    'encoder_dim': 2048,      
    'dropout': 0.5,
    
    # Training Strategy
    'batch_size': 64,        
    'num_epochs': 20,
    'fine_tune_encoder': True,
    
    # Learning Rates
    'learning_rate': 4e-4,    
    'encoder_lr': 1e-4,       
    
    # Regularization
    'weight_decay': 1e-5,
    'grad_clip': 5.0,
    'alpha_c': 1.0,          
    
    # M4 Pro Optimization
    'num_workers': 4,         
}

