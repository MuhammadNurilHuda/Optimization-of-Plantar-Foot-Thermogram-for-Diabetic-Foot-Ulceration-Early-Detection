# test_run.py

import yaml
import shutil
import os

def create_test_config():
    """Create a test config with reduced parameters for quick testing"""
    
    # Load original config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for test run
    config['train']['epochs'] = 2  # Reduce epochs
    config['train']['batch_size'] = 16  # Increase batch size for speed
    
    # Reduce enhancement parameters for testing
    config['data']['enhancements'] = {
        'CLAHE': ["2.0_(8, 8)"],  # Only 1 parameter
        'Gamma': ["0.5"],          # Only 1 parameter
    }
    
    # Use only 2 models
    config['model']['architectures'] = ['model1', 'model2']
    
    # Save test config
    with open('configs/test_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("Test config created: configs/test_config.yaml")
    print("Total experiments in test run: 2 enhancements × 1 param × 2 models = 4 experiments")
    
    return 'configs/test_config.yaml'

if __name__ == "__main__":
    test_config = create_test_config()
    print(f"\nRun test with: python main.py --config {test_config}")