#!/usr/bin/env python3
"""
Main CLI entry point for the HMM Market Regime Detection system.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        description='HMM Market Regime Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  ingest      Fetch market data from Alpha Vantage API
  train       Train HMM models with rolling windows
  predict     Make regime predictions on new data
  validate    Validate HMM predictions against historical data
  
Examples:
  python hmm_cli.py ingest --symbols EUR/USD USD/JPY
  python hmm_cli.py train --symbol EUR/USD --window-size 252
  python hmm_cli.py predict --symbol EUR/USD --data-file data.csv
  python hmm_cli.py validate --model-path models/latest.joblib --data-file data.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Fetch market data')
    ingest_parser.add_argument('--symbols', nargs='+', required=True,
                              help='Trading symbols to fetch')
    ingest_parser.add_argument('--output-dir', default='data/raw',
                              help='Output directory for data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train HMM models')
    train_parser.add_argument('--symbol', required=True, help='Trading symbol')
    train_parser.add_argument('--data-file', help='Data file path')
    train_parser.add_argument('--window-size', type=int, default=252,
                             help='Training window size')
    train_parser.add_argument('--step-size', type=int, default=21,
                             help='Rolling step size')
    train_parser.add_argument('--n-states', type=int, default=3,
                             help='Number of HMM states')
    train_parser.add_argument('--output-dir', default='models',
                             help='Output directory for models')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model-path', help='Path to model file')
    predict_parser.add_argument('--symbol', help='Symbol (to find latest model)')
    predict_parser.add_argument('--data-file', required=True,
                               help='Data file for prediction')
    predict_parser.add_argument('--output-file', help='Save predictions to file')
    
    # Validate command  
    validate_parser = subparsers.add_parser('validate', help='Validate predictions')
    validate_parser.add_argument('--model-path', required=True,
                                help='Path to trained model')
    validate_parser.add_argument('--data-file', required=True,
                                help='Data file for validation')
    validate_parser.add_argument('--output-file', help='Save validation results to file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'ingest':
            from ingest import main as ingest_main
            sys.argv = ['ingest.py', '--symbols'] + args.symbols + ['--output-dir', args.output_dir]
            ingest_main()
            
        elif args.command == 'train':
            from rolling_train import main as train_main
            sys.argv = [
                'rolling_train.py',
                '--symbol', args.symbol,
                '--window-size', str(args.window_size),
                '--step-size', str(args.step_size),
                '--n-states', str(args.n_states),
                '--output-dir', args.output_dir
            ]
            if args.data_file:
                sys.argv.extend(['--data-source', 'file', '--data-path', args.data_file])
            else:
                sys.argv.extend(['--data-source', 'api'])
            train_main()
            
        elif args.command == 'predict':
            from inference import main as inference_main
            sys.argv = ['inference.py', '--data-file', args.data_file]
            if args.model_path:
                sys.argv.extend(['--model-path', args.model_path])
            else:
                sys.argv.extend(['--symbol', args.symbol])
            if args.output_file:
                sys.argv.extend(['--output-file', args.output_file])
            inference_main()
            
        elif args.command == 'validate':
            from validate_predictions import main as validate_main
            sys.argv = [
                'validate_predictions.py',
                '--model-path', args.model_path,
                '--data-file', args.data_file
            ]
            if args.output_file:
                sys.argv.extend(['--output-file', args.output_file])
            validate_main()
            
    except Exception as e:
        print(f"Error executing {args.command}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
