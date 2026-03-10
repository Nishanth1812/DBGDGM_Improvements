"""
Main Preprocessing Orchestrator
================================

Unified interface to preprocess both OASIS and ADNI datasets.
"""

import argparse
from pathlib import Path
from typing import Optional
import json
import yaml

from src.oasis_processor import OASISDatasetProcessor
from src.adni_processor import ADNIDatasetProcessor


def load_config(config_path: str = 'config/preprocessing_config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def process_oasis(data_dir: str, output_dir: str, config: dict, verbose: bool = True) -> dict:
    """
    Process OASIS dataset.
    
    Parameters
    ----------
    data_dir : str
        OASIS dataset root directory
    output_dir : str
        Output directory for preprocessed data
    config : dict
        Configuration dictionary
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Processing results and statistics
    """
    print("="*60)
    print("OASIS Dataset Preprocessing")
    print("="*60)
    
    processor = OASISDatasetProcessor(data_dir, output_dir, verbose=verbose)
    
    # Discover subjects
    subjects = processor.discover_subjects()
    print(f"Discovered {len(subjects)} subjects")
    
    # Process all
    summary_df = processor.process_all_subjects()
    
    results = {
        'dataset': 'OASIS',
        'n_subjects': len(subjects),
        'n_processed': len(processor.processed_subjects),
        'output_dir': output_dir,
        'summary': summary_df.to_dict()
    }
    
    return results


def process_adni(data_dir: str, output_dir: str, config: dict, verbose: bool = True) -> dict:
    """
    Process ADNI dataset.
    
    Parameters
    ----------
    data_dir : str
        ADNI dataset root directory
    output_dir : str
        Output directory for preprocessed data
    config : dict
        Configuration dictionary
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Processing results and statistics
    """
    print("\n" + "="*60)
    print("ADNI Dataset Preprocessing")
    print("="*60)
    
    processor = ADNIDatasetProcessor(data_dir, output_dir, verbose=verbose)
    
    # Discover subjects
    subjects = processor.discover_subjects()
    print(f"Discovered {len(subjects)} subjects")
    
    # Process all
    summary_df = processor.process_all_subjects()
    
    results = {
        'dataset': 'ADNI',
        'n_subjects': len(subjects),
        'n_processed': len(processor.processed_subjects),
        'output_dir': output_dir,
        'summary': summary_df.to_dict()
    }
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Preprocess OASIS and ADNI datasets for DBGDGM model'
    )
    
    parser.add_argument('--dataset', type=str, choices=['oasis', 'adni', 'both'],
                       default='both', help='Dataset to process')
    parser.add_argument('--oasis-dir', type=str, help='OASIS dataset root directory')
    parser.add_argument('--adni-dir', type=str, help='ADNI dataset root directory')
    parser.add_argument('--output-dir', type=str, default='./preprocessed_data',
                       help='Output directory for preprocessed data')
    parser.add_argument('--config', type=str, default='config/preprocessing_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Configuration file not found: {args.config}")
        config = {}
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Process OASIS
    if args.dataset in ['oasis', 'both']:
        if args.oasis_dir is None:
            print("Error: OASIS directory not provided. Use --oasis-dir")
        else:
            oasis_output = Path(args.output_dir) / 'OASIS'
            result = process_oasis(args.oasis_dir, str(oasis_output), config, args.verbose)
            all_results['OASIS'] = result
    
    # Process ADNI
    if args.dataset in ['adni', 'both']:
        if args.adni_dir is None:
            print("Error: ADNI directory not provided. Use --adni-dir")
        else:
            adni_output = Path(args.output_dir) / 'ADNI'
            result = process_adni(args.adni_dir, str(adni_output), config, args.verbose)
            all_results['ADNI'] = result
    
    # Save results summary
    results_path = Path(args.output_dir) / 'preprocessing_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Preprocessing Complete")
    print("="*60)
    print(f"Results saved to: {results_path}")
    
    return all_results


if __name__ == '__main__':
    main()
