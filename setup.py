#!/usr/bin/env python3
"""
Setup script for the HMM Market Regime Detection system.
"""

import os
import sys
from pathlib import Path
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def setup_environment():
    """Set up virtual environment and install dependencies."""
    print("\n=== Setting up environment ===")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if not in_venv:
        print("Warning: Not in a virtual environment. It's recommended to create one:")
        print("  python -m venv venv")
        print("  venv\\Scripts\\activate  (Windows)")
        print("  source venv/bin/activate  (Linux/Mac)")
        print("\nContinuing with global installation...")
    
    # Install dependencies
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("You may need to install some dependencies manually:")
        print("  pip install numpy pandas scikit-learn matplotlib joblib requests")
        print("  pip install pomegranate  # or hmmlearn")

def setup_directories():
    """Create necessary directories."""
    print("\n=== Setting up directories ===")
    
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'results',
        'logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def setup_config():
    """Set up configuration files."""
    print("\n=== Setting up configuration ===")
    
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✓ Created .env file from template")
        print("  Please edit .env and add your Alpha Vantage API key")
    else:
        print("✓ Configuration files already exist")

def run_tests():
    """Run basic tests to verify installation."""
    print("\n=== Running basic tests ===")
    
    try:
        # Test imports
        sys.path.insert(0, str(Path('src')))
        
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        print("✓ Core dependencies available")
        
        # Test HMM backend
        try:
            import pomegranate
            print("✓ Pomegranate backend available")
        except ImportError:
            try:
                import hmmlearn
                print("✓ hmmlearn backend available")
            except ImportError:
                print("⚠ No HMM backend found. Please install pomegranate or hmmlearn:")
                print("  pip install pomegranate")
                print("  # OR")
                print("  pip install hmmlearn")
        
        # Test our modules
        from features import compute_features
        from model_wrapper import HMMRegimeModel
        from utils import setup_logging
        print("✓ HMM system modules loaded successfully")
        
        # Quick functionality test
        np.random.seed(42)
        test_data = pd.DataFrame({
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 101,
            'Low': np.random.randn(100) + 99,
            'Close': np.random.randn(100) + 100
        }, index=pd.date_range('2020-01-01', periods=100))
        
        features = compute_features(test_data)
        print(f"✓ Feature computation working (shape: {features.shape})")
        
    except Exception as e:
        print(f"⚠ Test failed: {e}")
        print("Some components may not work correctly.")

def print_usage_examples():
    """Print usage examples."""
    print("\n=== Usage Examples ===")
    print("""
1. Fetch market data:
   python hmm_cli.py ingest --symbols EUR/USD USD/JPY

2. Train HMM models:
   python hmm_cli.py train --symbol EUR/USD --window-size 252

3. Make predictions:
   python hmm_cli.py predict --symbol EUR/USD --data-file data/raw/EUR_USD/latest.csv

4. Run backtest:
   python hmm_cli.py backtest --model-path models/latest.joblib --data-file data.csv

5. Interactive analysis:
   jupyter notebook notebooks/hmm_eda_demo.ipynb

6. Run tests:
   python -m pytest tests/ -v
""")

def main():
    """Main setup function."""
    print("HMM Market Regime Detection System - Setup")
    print("=" * 50)
    
    check_python_version()
    setup_directories() 
    setup_environment()
    setup_config()
    run_tests()
    print_usage_examples()
    
    print("\n=== Setup Complete ===")
    print("✓ System is ready to use!")
    print("\nDon't forget to:")
    print("1. Add your Alpha Vantage API key to .env file")
    print("2. Activate your virtual environment if using one")
    print("3. Check the README.md for detailed instructions")

if __name__ == '__main__':
    main()
