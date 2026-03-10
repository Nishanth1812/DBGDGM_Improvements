"""
Metadata Generator and Experiments Manager

Tools for:
1. Creating and validating metadata CSV files
2. Managing experiment runs and results
3. Generating baseline configurations
"""

import os
import csv
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import argparse


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

class MetadataManager:
    """Create and validate metadata files for MM-DBGDGM training."""
    
    LABEL_MAP = {
        'CN': 0,
        'eMCI': 1,
        'lMCI': 2,
        'AD': 3,
    }
    
    LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
    
    @staticmethod
    def create_metadata_csv(
        output_file: str,
        subjects: List[Dict],
        validate: bool = True
    ):
        """
        Create metadata CSV file.
        
        Args:
            output_file: Path to output CSV
            subjects: List of dicts with keys: subject_id, timepoint, label
            validate: Validate before writing
        
        Example:
            >>> subjects = [
            ...     {'subject_id': 'ADNI_001', 'timepoint': '2024_01_15', 'label': 0},
            ...     {'subject_id': 'ADNI_002', 'timepoint': '2024_02_20', 'label': 1},
            ... ]
            >>> MetadataManager.create_metadata_csv('train.csv', subjects)
        """
        
        if validate:
            MetadataManager.validate_subjects(subjects)
        
        # Ensure label is integer
        for s in subjects:
            if isinstance(s['label'], str):
                s['label'] = MetadataManager.LABEL_MAP[s['label']]
        
        # Write CSV
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['subject_id', 'timepoint', 'label']
            )
            writer.writeheader()
            writer.writerows(subjects)
        
        print(f"✓ Created metadata file: {output_file}")
        print(f"  Samples: {len(subjects)}")
        print(f"  Label distribution:")
        
        label_counts = {}
        for s in subjects:
            label = s['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label in sorted(label_counts.keys()):
            name = MetadataManager.LABEL_NAMES[label]
            count = label_counts[label]
            pct = 100 * count / len(subjects)
            print(f"    {name}: {count} ({pct:.1f}%)")
    
    @staticmethod
    def validate_subjects(subjects: List[Dict]):
        """Validate subject list."""
        required_fields = {'subject_id', 'timepoint', 'label'}
        
        for i, s in enumerate(subjects):
            # Check required fields
            missing = required_fields - set(s.keys())
            if missing:
                raise ValueError(f"Subject {i} missing fields: {missing}")
            
            # Check label is valid
            label = s['label']
            if isinstance(label, str):
                if label not in MetadataManager.LABEL_MAP:
                    raise ValueError(
                        f"Invalid label '{label}' for subject {i}. "
                        f"Must be one of: {list(MetadataManager.LABEL_MAP.keys())}"
                    )
            elif isinstance(label, int):
                if label not in MetadataManager.LABEL_NAMES:
                    raise ValueError(
                        f"Invalid label '{label}' for subject {i}. "
                        f"Must be one of: {list(MetadataManager.LABEL_NAMES.keys())}"
                    )
            else:
                raise ValueError(f"Label must be string or int, got {type(label)}")
    
    @staticmethod
    def validate_metadata_file(
        csv_file: str,
        dataset_root: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate metadata CSV file.
        
        Args:
            csv_file: Path to metadata CSV
            dataset_root: If provided, check data files exist
        
        Returns:
            (is_valid, warnings)
        """
        warnings = []
        
        # Check file exists
        if not os.path.exists(csv_file):
            return False, [f"File not found: {csv_file}"]
        
        # Load and check structure
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            return False, [f"Failed to read CSV: {e}"]
        
        required_cols = {'subject_id', 'timepoint', 'label'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            return False, [f"Missing columns: {missing}"]
        
        # Check label values
        for i, label in enumerate(df['label']):
            if label not in MetadataManager.LABEL_NAMES:
                warnings.append(f"Row {i+2}: Invalid label {label}")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['subject_id', 'timepoint'])
        if duplicates.any():
            warnings.append(f"Found {duplicates.sum()} duplicate subject/timepoint pairs")
        
        # Check data files if dataset_root provided
        if dataset_root:
            for i, row in df.iterrows():
                fmri_file = os.path.join(
                    dataset_root, 'fmri',
                    row['subject_id'], row['timepoint'],
                    'fmri_windows_dbgdgm.npy'
                )
                smri_file = os.path.join(
                    dataset_root, 'smri',
                    row['subject_id'], row['timepoint'],
                    'features.npy'
                )
                
                if not os.path.exists(fmri_file):
                    warnings.append(f"Row {i+2}: fMRI file not found: {fmri_file}")
                if not os.path.exists(smri_file):
                    warnings.append(f"Row {i+2}: sMRI file not found: {smri_file}")
        
        return len(warnings) == 0, warnings
    
    @staticmethod
    def generate_example_metadata(
        output_dir: str = '.',
        n_train: int = 600,
        n_val: int = 150,
        n_test: int = 150,
    ):
        """
        Generate example metadata files for testing.
        
        Args:
            output_dir: Directory to save CSV files
            n_train: Number of training samples
            n_val: Number of validation samples
            n_test: Number of test samples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Distribution: roughly equal across classes
        def create_subjects(n_samples, start_id=0):
            subjects = []
            for i in range(n_samples):
                label = (i % 4)  # Cycle through 4 classes
                subjects.append({
                    'subject_id': f'ADNI_{start_id + i + 1:04d}',
                    'timepoint': f'2024_{(i % 12) + 1:02d}_{(i % 28) + 1:02d}',
                    'label': label,
                })
            return subjects
        
        train_subjects = create_subjects(n_train, 0)
        val_subjects = create_subjects(n_val, n_train)
        test_subjects = create_subjects(n_test, n_train + n_val)
        
        # Write files
        MetadataManager.create_metadata_csv(
            os.path.join(output_dir, 'train_metadata.csv'),
            train_subjects,
            validate=False
        )
        MetadataManager.create_metadata_csv(
            os.path.join(output_dir, 'val_metadata.csv'),
            val_subjects,
            validate=False
        )
        MetadataManager.create_metadata_csv(
            os.path.join(output_dir, 'test_metadata.csv'),
            test_subjects,
            validate=False
        )
        
        print(f"✓ Generated example metadata files in {output_dir}")


# ============================================================================
# EXPERIMENT MANAGEMENT
# ============================================================================

class ExperimentManager:
    """Manage experiment runs and results."""
    
    @staticmethod
    def list_experiments(experiments_dir: str = './experiments') -> List[Dict]:
        """
        List all experiment runs.
        
        Args:
            experiments_dir: Directory containing experiments
        
        Returns:
            List of experiment dicts with metadata
        """
        experiments = []
        
        if not os.path.exists(experiments_dir):
            return experiments
        
        for exp_dir in sorted(os.listdir(experiments_dir), reverse=True):
            exp_path = os.path.join(experiments_dir, exp_dir)
            
            if not os.path.isdir(exp_path):
                continue
            
            # Try to load results
            results_file = os.path.join(exp_path, 'results.json')
            config_file = os.path.join(exp_path, 'config.json')
            
            exp_info = {
                'name': exp_dir,
                'path': exp_path,
                'timestamp': exp_dir.split('_')[-1] if '_' in exp_dir else 'unknown',
            }
            
            if os.path.exists(results_file):
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    exp_info['val_acc'] = results.get('val_metrics', {}).get('accuracy')
                    exp_info['test_acc'] = results.get('test_metrics', {}).get('accuracy')
                except:
                    pass
            
            if os.path.exists(config_file):
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                    exp_info['batch_size'] = config.get('batch_size')
                    exp_info['learning_rate'] = config.get('learning_rate')
                except:
                    pass
            
            experiments.append(exp_info)
        
        return experiments
    
    @staticmethod
    def print_experiments(experiments: List[Dict]):
        """Print formatted experiment list."""
        if not experiments:
            print("No experiments found.")
            return
        
        print("\nRecent Experiments:")
        print("=" * 100)
        print(f"{'Experiment':<30} {'Val Acc':<10} {'Test Acc':<10} {'Batch':<8} {'LR':<10}")
        print("=" * 100)
        
        for exp in experiments[:10]:  # Show top 10
            name = exp['name'][:28]
            val_acc = f"{exp.get('val_acc', float('nan')):.4f}" if exp.get('val_acc') else 'N/A'
            test_acc = f"{exp.get('test_acc', float('nan')):.4f}" if exp.get('test_acc') else 'N/A'
            batch = str(exp.get('batch_size', 'N/A'))
            lr = str(exp.get('learning_rate', 'N/A'))
            
            print(f"{name:<30} {val_acc:<10} {test_acc:<10} {batch:<8} {lr:<10}")
        
        print("=" * 100)
    
    @staticmethod
    def compare_experiments(
        experiments_dir: str = './experiments',
        metric: str = 'val_acc',
        top_n: int = 5,
    ):
        """
        Compare top experiments by metric.
        
        Args:
            experiments_dir: Directory containing experiments
            metric: Metric to sort by ('val_acc', 'test_acc')
            top_n: Number of top experiments to show
        """
        experiments = ExperimentManager.list_experiments(experiments_dir)
        
        # Filter experiments with the metric
        valid_exps = [e for e in experiments if metric in e]
        
        if not valid_exps:
            print(f"No experiments with metric '{metric}' found.")
            return
        
        # Sort by metric
        valid_exps = sorted(valid_exps, key=lambda x: x[metric], reverse=True)
        
        print(f"\nTop {min(top_n, len(valid_exps))} Experiments by {metric}:")
        print("=" * 100)
        
        for i, exp in enumerate(valid_exps[:top_n], 1):
            print(f"{i}. {exp['name']}")
            print(f"   {metric}: {exp[metric]:.4f}")
            print(f"   Batch: {exp.get('batch_size')}, LR: {exp.get('learning_rate')}")
            print()


# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

class ConfigTemplates:
    """Generate configuration templates for different scenarios."""
    
    @staticmethod
    def get_default_config() -> Dict:
        """Default configuration."""
        return {
            'data': {
                'dataset_root': '/path/to/preprocessed_data',
                'train_metadata': '/path/to/train_metadata.csv',
                'val_metadata': '/path/to/val_metadata.csv',
                'test_metadata': '/path/to/test_metadata.csv',
                'normalize': True,
            },
            'model': {
                'n_roi': 200,
                'seq_len': 50,
                'n_smri_features': 5,
                'latent_dim': 256,
                'num_classes': 4,
                'use_gat_encoder': True,
                'use_attention_fusion': True,
                'dropout': 0.1,
            },
            'training': {
                'batch_size': 32,
                'num_workers': 4,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'patience': 10,
                'annealing_epochs': 20,
            },
            'loss_weights': {
                'lambda_kl': 0.1,
                'lambda_align': 0.1,
                'lambda_recon': 0.1,
            },
        }
    
    @staticmethod
    def get_lightweight_config() -> Dict:
        """Configuration for limited GPU memory."""
        cfg = ConfigTemplates.get_default_config()
        cfg['model'].update({
            'latent_dim': 128,
            'use_gat_encoder': False,
            'use_attention_fusion': False,
        })
        cfg['training'].update({
            'batch_size': 16,
            'num_workers': 2,
        })
        return cfg
    
    @staticmethod
    def get_fast_config() -> Dict:
        """Configuration for quick experiments."""
        cfg = ConfigTemplates.get_default_config()
        cfg['training'].update({
            'num_epochs': 10,
            'patience': 3,
        })
        return cfg
    
    @staticmethod
    def save_config(config: Dict, output_file: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved configuration to {output_file}")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Command-line interface for metadata and experiment management."""
    parser = argparse.ArgumentParser(
        description='MM-DBGDGM Metadata and Experiment Management'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Metadata commands
    metadata = subparsers.add_parser('metadata', help='Metadata management')
    metadata.add_argument('--validate', type=str, help='Validate metadata CSV file')
    metadata.add_argument('--dataset-root', type=str, help='Dataset root directory (for file validation)')
    metadata.add_argument('--generate-example', type=str, help='Generate example metadata files')
    metadata.add_argument('--n-train', type=int, default=600)
    metadata.add_argument('--n-val', type=int, default=150)
    metadata.add_argument('--n-test', type=int, default=150)
    
    # Experiment commands
    experiments = subparsers.add_parser('experiments', help='Experiment management')
    experiments.add_argument('--list', action='store_true', help='List experiments')
    experiments.add_argument('--compare', action='store_true', help='Compare top experiments')
    experiments.add_argument('--metric', type=str, default='val_acc', help='Metric to compare')
    experiments.add_argument('--top', type=int, default=5, help='Top N experiments')
    experiments.add_argument('--dir', type=str, default='./experiments', help='Experiments directory')
    
    # Config commands
    config = subparsers.add_parser('config', help='Configuration management')
    config.add_argument('--save', type=str, help='Save configuration template')
    config.add_argument('--template', type=str, 
                       choices=['default', 'lightweight', 'fast'],
                       default='default',
                       help='Configuration template to use')
    
    args = parser.parse_args()
    
    # Metadata operations
    if args.command == 'metadata':
        if args.validate:
            is_valid, warnings = MetadataManager.validate_metadata_file(
                args.validate,
                dataset_root=args.dataset_root
            )
            
            if is_valid:
                print(f"✓ Metadata file is valid: {args.validate}")
            else:
                print(f"✗ Metadata file has errors: {args.validate}")
            
            for warning in warnings:
                print(f"  ⚠ {warning}")
        
        elif args.generate_example:
            MetadataManager.generate_example_metadata(
                output_dir=args.generate_example,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
            )
    
    # Experiment operations
    elif args.command == 'experiments':
        if args.list:
            experiments = ExperimentManager.list_experiments(args.dir)
            ExperimentManager.print_experiments(experiments)
        
        elif args.compare:
            ExperimentManager.compare_experiments(
                experiments_dir=args.dir,
                metric=args.metric,
                top_n=args.top,
            )
    
    # Config operations
    elif args.command == 'config':
        if args.template == 'default':
            cfg = ConfigTemplates.get_default_config()
        elif args.template == 'lightweight':
            cfg = ConfigTemplates.get_lightweight_config()
        elif args.template == 'fast':
            cfg = ConfigTemplates.get_fast_config()
        
        if args.save:
            ConfigTemplates.save_config(cfg, args.save)
        else:
            print(json.dumps(cfg, indent=2))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
