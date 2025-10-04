#!/usr/bin/env python3
"""
Automated HMM Market Regime Detection Workflow
Streamlines the entire process: ingest → train → predict → backtest
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
from typing import List, Optional

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a colored header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_step(step_num: int, text: str):
    """Print a step indicator."""
    print(f"{Colors.OKCYAN}{Colors.BOLD}[Step {step_num}] {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command to run as list of strings
        description: Description of what the command does
        
    Returns:
        True if successful, False otherwise
    """
    print_info(f"Running: {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print_success(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"{description} failed: {str(e)}")
        return False


def get_latest_file(pattern: str) -> Optional[Path]:
    """
    Get the most recent file matching the pattern.
    
    Args:
        pattern: Glob pattern to match files
        
    Returns:
        Path to latest file or None if not found
    """
    files = list(Path('.').glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def select_symbols() -> List[str]:
    """
    Interactive symbol selection.
    
    Returns:
        List of selected symbols
    """
    print_header("Symbol Selection")
    
    popular_symbols = [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF",
        "NZD/USD", "USD/CAD", "EUR/GBP", "EUR/JPY", "GBP/JPY"
    ]
    
    print("Popular Currency Pairs:")
    for i, symbol in enumerate(popular_symbols, 1):
        print(f"  {i:2d}. {symbol}")
    
    print("\nOptions:")
    print("  - Enter numbers (e.g., '1 2 3' for EUR/USD, GBP/USD, USD/JPY)")
    print("  - Enter 'all' for all popular pairs")
    print("  - Enter custom symbol (e.g., 'BTC/USD')")
    print("  - Press Enter for EUR/USD (default)")
    
    choice = input("\n> ").strip()
    
    if not choice:
        print_info("Using default: EUR/USD")
        return ["EUR/USD"]
    
    if choice.lower() == 'all':
        print_success(f"Selected all {len(popular_symbols)} popular pairs")
        return popular_symbols
    
    # Try to parse as numbers
    try:
        indices = [int(x) for x in choice.split()]
        selected = [popular_symbols[i-1] for i in indices if 1 <= i <= len(popular_symbols)]
        if selected:
            print_success(f"Selected: {', '.join(selected)}")
            return selected
    except ValueError:
        pass
    
    # Treat as custom symbol
    symbols = [s.strip() for s in choice.split()]
    print_success(f"Using custom symbols: {', '.join(symbols)}")
    return symbols


def confirm_parameters(symbols: List[str], window_size: int, step_size: int, 
                       n_states: int, strategies: List[str]) -> bool:
    """
    Display parameters and ask for confirmation.
    
    Returns:
        True if user confirms, False otherwise
    """
    print_header("Workflow Configuration")
    
    print(f"{Colors.BOLD}Symbols:{Colors.ENDC}")
    for symbol in symbols:
        print(f"  • {symbol}")
    
    print(f"\n{Colors.BOLD}Training Parameters:{Colors.ENDC}")
    print(f"  • Window Size:  {window_size} days (~{window_size/252:.1f} years)")
    print(f"  • Step Size:    {step_size} days (~{step_size/21:.1f} months)")
    print(f"  • HMM States:   {n_states} (Bull/Bear/Sideways)")
    
    print(f"\n{Colors.BOLD}Backtest Strategies:{Colors.ENDC}")
    for strategy in strategies:
        print(f"  • {strategy.capitalize()}")
    
    print(f"\n{Colors.BOLD}Workflow Steps:{Colors.ENDC}")
    print("  1. Ingest market data from Alpha Vantage API")
    print("  2. Train HMM models with rolling windows")
    print("  3. Generate regime predictions")
    print("  4. Run backtests for each strategy")
    print("  5. Save all results to JSON files")
    
    response = input(f"\n{Colors.WARNING}Proceed with this configuration? (Y/n): {Colors.ENDC}").strip().lower()
    return response in ['', 'y', 'yes']


def run_workflow(symbols: List[str], window_size: int = 252, step_size: int = 21,
                 n_states: int = 3, strategies: List[str] = None,
                 skip_ingest: bool = False, skip_train: bool = False) -> dict:
    """
    Run the complete HMM workflow.
    
    Args:
        symbols: List of trading symbols
        window_size: Training window size in days
        step_size: Rolling window step size in days
        n_states: Number of HMM states
        strategies: List of backtest strategies
        skip_ingest: Skip data ingestion step
        skip_train: Skip model training step
        
    Returns:
        Dictionary with workflow results
    """
    if strategies is None:
        strategies = ['simple', 'momentum', 'adaptive']
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'symbols': symbols,
        'parameters': {
            'window_size': window_size,
            'step_size': step_size,
            'n_states': n_states
        },
        'steps_completed': [],
        'predictions': {},
        'backtests': {}
    }
    
    print_header("🚀 Starting HMM Workflow")
    
    # Step 1: Ingest data
    if not skip_ingest:
        print_step(1, "Ingesting Market Data")
        cmd = ['python', 'hmm_cli.py', 'ingest', '--symbols'] + symbols
        if not run_command(cmd, "Data ingestion"):
            print_error("Failed to ingest data. Aborting.")
            return results
        results['steps_completed'].append('ingest')
        print()
    else:
        print_warning("Skipping data ingestion (using existing data)")
    
    # Process each symbol
    for i, symbol in enumerate(symbols, 1):
        symbol_clean = symbol.replace('/', '_')
        
        print_header(f"Processing Symbol {i}/{len(symbols)}: {symbol}")
        
        # Find latest data file
        data_pattern = f"data/raw/{symbol_clean}/*.parquet"
        latest_data = get_latest_file(data_pattern)
        
        if not latest_data:
            print_error(f"No data file found for {symbol}. Skipping.")
            continue
        
        print_info(f"Using data file: {latest_data}")
        
        # Step 2: Train model
        if not skip_train:
            print_step(2, f"Training HMM Model for {symbol}")
            cmd = [
                'python', 'hmm_cli.py', 'train',
                '--symbol', symbol_clean,
                '--data-file', str(latest_data),
                '--window-size', str(window_size),
                '--step-size', str(step_size),
                '--n-states', str(n_states)
            ]
            if not run_command(cmd, f"Model training for {symbol}"):
                print_error(f"Failed to train model for {symbol}. Skipping.")
                continue
            results['steps_completed'].append(f'train_{symbol_clean}')
            print()
        else:
            print_warning(f"Skipping model training for {symbol} (using existing model)")
        
        # Find latest model
        model_pattern = f"models/hmm_{symbol_clean}_*.joblib"
        latest_model = get_latest_file(model_pattern)
        
        if not latest_model:
            print_error(f"No model file found for {symbol}. Skipping.")
            continue
        
        print_info(f"Using model file: {latest_model}")
        
        # Step 3: Make predictions
        print_step(3, f"Generating Predictions for {symbol}")
        predictions_file = f"predictions_{symbol_clean}.json"
        cmd = [
            'python', 'hmm_cli.py', 'predict',
            '--model-path', str(latest_model),
            '--data-file', str(latest_data),
            '--output-file', predictions_file
        ]
        if run_command(cmd, f"Prediction generation for {symbol}"):
            results['steps_completed'].append(f'predict_{symbol_clean}')
            results['predictions'][symbol] = predictions_file
            
            # Load and display prediction summary
            try:
                with open(predictions_file, 'r') as f:
                    pred_data = json.load(f)
                    regime = pred_data['current_regime']['regime_label']
                    confidence = pred_data['current_regime']['posterior_probability'] * 100
                    print_success(f"Current Regime: {regime} ({confidence:.1f}% confidence)")
            except Exception as e:
                print_warning(f"Could not read predictions: {e}")
        print()
        
        # Step 4: Run backtests
        results['backtests'][symbol] = {}
        
        for j, strategy in enumerate(strategies, 1):
            print_step(4, f"Running Backtest for {symbol} - {strategy.capitalize()} Strategy ({j}/{len(strategies)})")
            backtest_file = f"backtest_{symbol_clean}_{strategy}.json"
            cmd = [
                'python', 'hmm_cli.py', 'backtest',
                '--model-path', str(latest_model),
                '--data-file', str(latest_data),
                '--strategy', strategy,
                '--output-file', backtest_file
            ]
            if run_command(cmd, f"{strategy.capitalize()} backtest for {symbol}"):
                results['steps_completed'].append(f'backtest_{symbol_clean}_{strategy}')
                results['backtests'][symbol][strategy] = backtest_file
                
                # Load and display backtest summary
                try:
                    with open(backtest_file, 'r') as f:
                        bt_data = json.load(f)
                        total_return = bt_data['performance']['total_return']
                        sharpe = bt_data['performance']['sharpe_ratio']
                        print_success(f"Total Return: {total_return:+.2f}% | Sharpe: {sharpe:.2f}")
                except Exception as e:
                    print_warning(f"Could not read backtest results: {e}")
            print()
    
    # Save workflow summary
    summary_file = 'workflow_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_header("✅ Workflow Complete!")
    print_success(f"Processed {len(symbols)} symbol(s)")
    print_success(f"Completed {len(results['steps_completed'])} steps")
    print_info(f"Summary saved to: {summary_file}")
    
    print(f"\n{Colors.BOLD}Generated Files:{Colors.ENDC}")
    for symbol in symbols:
        symbol_clean = symbol.replace('/', '_')
        print(f"\n  {Colors.BOLD}{symbol}:{Colors.ENDC}")
        if symbol in results['predictions']:
            print(f"    • Predictions: {results['predictions'][symbol]}")
        if symbol in results['backtests']:
            for strategy, filepath in results['backtests'][symbol].items():
                print(f"    • Backtest ({strategy}): {filepath}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automated HMM Market Regime Detection Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (asks for symbols)
  python doit.py
  
  # Automatic mode with single symbol
  python doit.py --symbols EUR/USD
  
  # Multiple symbols with custom parameters
  python doit.py --symbols EUR/USD GBP/USD --window-size 180 --n-states 4
  
  # Skip ingestion and training (use existing data/models)
  python doit.py --symbols EUR/USD --skip-ingest --skip-train
  
  # Run only simple backtest strategy
  python doit.py --symbols EUR/USD --strategies simple
  
  # Non-interactive mode (no confirmation)
  python doit.py --symbols EUR/USD --yes
        """
    )
    
    parser.add_argument('--symbols', nargs='+', help='Trading symbols (e.g., EUR/USD GBP/USD)')
    parser.add_argument('--window-size', type=int, default=252, 
                       help='Training window size in days (default: 252)')
    parser.add_argument('--step-size', type=int, default=21,
                       help='Rolling window step size in days (default: 21)')
    parser.add_argument('--n-states', type=int, default=3,
                       help='Number of HMM states (default: 3)')
    parser.add_argument('--strategies', nargs='+', 
                       choices=['simple', 'momentum', 'adaptive'],
                       default=['simple', 'momentum', 'adaptive'],
                       help='Backtest strategies to run (default: all)')
    parser.add_argument('--skip-ingest', action='store_true',
                       help='Skip data ingestion (use existing data)')
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip model training (use existing models)')
    parser.add_argument('-y', '--yes', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    print_header("🎯 HMM Market Regime Detection - Automated Workflow")
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print_info(f"Using command-line symbols: {', '.join(symbols)}")
    else:
        symbols = select_symbols()
    
    # Confirm parameters
    if not args.yes:
        if not confirm_parameters(symbols, args.window_size, args.step_size, 
                                  args.n_states, args.strategies):
            print_warning("Workflow cancelled by user")
            return
    
    # Run workflow
    try:
        results = run_workflow(
            symbols=symbols,
            window_size=args.window_size,
            step_size=args.step_size,
            n_states=args.n_states,
            strategies=args.strategies,
            skip_ingest=args.skip_ingest,
            skip_train=args.skip_train
        )
        
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}All done! Happy trading! 🚀{Colors.ENDC}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Workflow interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
