# main.py

import argparse
import logging.config
import os
from src.experiment_runner import run_full_grid_experiment

def main():
    parser = argparse.ArgumentParser(description='Run DFU Detection Grid Search Experiments')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--test-run', action='store_true',
                        help='Run dengan epochs kecil untuk testing')
    args = parser.parse_args()
    
    # Setup logging
    logging.config.fileConfig('configs/logging.conf')
    
    # Create necessary directories
    directories = [
        'logs',
        'logs/tfboard',
        'logs/history_csv',
        'logs/evaluasi_csv',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("="*60)
    print("DFU Detection Grid Search Experiment")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Test run: {args.test_run}")
    
    if args.test_run:
        print("\n⚠️  Running in TEST MODE - reduced epochs for quick testing")
    
    print("\nStarting experiments...")
    print("Check logs/ directory for detailed progress")
    print("="*60)
    
    # Run experiments
    run_full_grid_experiment(args.config)

if __name__ == "__main__":
    main()