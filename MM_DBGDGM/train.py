"""
Complete training script for MM-DBGDGM.
Usage: python train.py --config configs/default.yaml
"""

import argparse
import torch
import logging
from pathlib import Path
from datetime import datetime
import yaml

try:
    from .models import MM_DBGDGM
    from .training.losses import MM_DBGDGM_Loss
    from .training.trainer import Trainer
    from .data.dataset import create_dataloaders
except ImportError:
    from models import MM_DBGDGM
    from training.losses import MM_DBGDGM_Loss
    from training.trainer import Trainer
    from data.dataset import create_dataloaders


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    output_dir = Path(f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_dataloaders(
        dataset_root=config['data']['dataset_root'],
        train_metadata=config['data']['train_metadata'],
        val_metadata=config['data']['val_metadata'],
        test_metadata=config['data'].get('test_metadata'),
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        normalize=config['data'].get('normalize', True)
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders.get('test')
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = MM_DBGDGM(
        n_roi=config['model']['n_roi'],
        seq_len=config['model']['seq_len'],
        n_smri_features=config['model']['n_smri_features'],
        latent_dim=config['model']['latent_dim'],
        num_classes=config['model']['num_classes'],
        use_gat_encoder=config['model'].get('use_gat_encoder', True),
        use_attention_fusion=config['model'].get('use_attention_fusion', True),
        num_fusion_heads=config['model'].get('num_fusion_heads', 4),
        num_fusion_iterations=config['model'].get('num_fusion_iterations', 2),
        dropout=config['model'].get('dropout', 0.1)
    ).to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create loss function
    logger.info("Creating loss function...")
    criterion = MM_DBGDGM_Loss(
        num_classes=config['model']['num_classes'],
        lambda_kl=config['training'].get('lambda_kl', 0.1),
        lambda_align=config['training'].get('lambda_align', 0.1),
        lambda_recon=config['training'].get('lambda_recon', 0.1)
    ).to(device)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        criterion=criterion,
        device=device,
        output_dir=str(output_dir),
        seed=config['training'].get('seed', 42)
    )
    
    # Train
    logger.info("Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5),
        patience=config['training'].get('patience', 10),
        annealing_epochs=config['training'].get('annealing_epochs', 20)
    )
    
    # Test (if test data available)
    if test_loader:
        logger.info("Loading best model for testing...")
        best_model_path = output_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning(f"Best model checkpoint not found at {best_model_path}; testing current model weights")
        
        logger.info("Running on test set...")
        test_metrics = trainer.validate(test_loader, epoch=0, beta_annealing=1.0)
        logger.info(f"Test Loss: {test_metrics['val_total']:.4f}, Test Acc: {test_metrics['val_accuracy']:.4f}")
        
        # Save test results
        with open(output_dir / 'test_results.yaml', 'w') as f:
            yaml.dump({k: float(v) for k, v in test_metrics.items()}, f)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


def create_default_config(output_path: str = 'configs/default.yaml'):
    """Create a default configuration file."""
    config = {
        'data': {
            'dataset_root': '/path/to/preprocessed_data',
            'train_metadata': '/path/to/train_metadata.csv',
            'val_metadata': '/path/to/val_metadata.csv',
            'test_metadata': '/path/to/test_metadata.csv',
            'normalize': True
        },
        'model': {
            'n_roi': 200,
            'seq_len': 50,
            'n_smri_features': 5,
            'latent_dim': 256,
            'num_classes': 4,
            'use_gat_encoder': True,
            'use_attention_fusion': True,
            'num_fusion_heads': 4,
            'num_fusion_iterations': 2,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'num_workers': 4,
            'num_epochs': 50,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'lambda_kl': 0.1,
            'lambda_align': 0.1,
            'lambda_recon': 0.1,
            'patience': 10,
            'annealing_epochs': 20,
            'seed': 42
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Default config created: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MM-DBGDGM model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a default configuration file and exit'
    )
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config(args.config)
    else:
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            logger.info("Run with --create-config to create a default config file")
            exit(1)
        
        main(args)
