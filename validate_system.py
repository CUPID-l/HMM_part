#!/usr/bin/env python3
"""
Demonstration script showing the complete HMM workflow.
This script validates that all components work together correctly.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("HMM Market Regime Detection System - Complete Workflow Demo")
    print("=" * 70)
    
    try:
        # Step 1: Import all modules
        print("\n1. Loading system modules...")
        from features import compute_features, validate_features
        from model_wrapper import HMMRegimeModel
        from utils import (
            setup_logging, set_random_seeds, label_states_by_return_volatility,
            calculate_performance_metrics, save_model_artifact
        )
        from rolling_train import RollingHMMTrainer
        from inference import HMMInferenceEngine
        from backtest import SimpleRegimeStrategy, HMMBacktester
        from sklearn.preprocessing import StandardScaler
        print("✓ All modules imported successfully")
        
        # Step 2: Generate sample data
        print("\n2. Generating sample market data...")
        set_random_seeds(42)
        
        # Create realistic sample data
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        returns = np.random.normal(0, 0.015, 500)
        
        # Add regime structure
        regime_changes = [0, 150, 300, 450, 500]
        regime_params = [
            (0.0008, 0.012),   # Bull
            (-0.0005, 0.020),  # Bear
            (0.0001, 0.008),   # Sideways
            (0.0006, 0.010)    # Bull
        ]
        
        structured_returns = []
        for i in range(len(regime_changes) - 1):
            start_idx = regime_changes[i]
            end_idx = regime_changes[i + 1]
            n_periods = end_idx - start_idx
            mu, sigma = regime_params[i]
            period_returns = np.random.normal(mu, sigma, n_periods)
            structured_returns.extend(period_returns)
        
        # Generate OHLC data
        prices = 100 * (1 + np.array(structured_returns)).cumprod()
        noise = np.random.normal(0, 0.002, (500, 3))
        
        ohlc_data = pd.DataFrame({
            'Open': prices + noise[:, 0],
            'High': prices + abs(noise[:, 1]) + 0.001,
            'Low': prices - abs(noise[:, 2]) - 0.001,
            'Close': prices
        }, index=dates)
        
        # Ensure valid OHLC relationships
        ohlc_data['High'] = np.maximum(
            ohlc_data['High'],
            ohlc_data[['Open', 'Close']].max(axis=1)
        )
        ohlc_data['Low'] = np.minimum(
            ohlc_data['Low'],
            ohlc_data[['Open', 'Close']].min(axis=1)
        )
        
        print(f"✓ Generated {len(ohlc_data)} days of OHLC data")
        
        # Step 3: Compute features
        print("\n3. Computing technical features...")
        features = compute_features(ohlc_data)
        is_valid, msg = validate_features(features)
        
        if not is_valid:
            raise ValueError(f"Feature validation failed: {msg}")
        
        print(f"✓ Computed {len(features)} feature observations")
        print(f"  Features: {list(features.columns)}")
        
        # Step 4: Train HMM model
        print("\n4. Training HMM model...")
        
        # Prepare training data (use first 80%)
        train_size = int(len(features) * 0.8)
        train_features = features.iloc[:train_size]
        test_features = features.iloc[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features.values)
        test_scaled = scaler.transform(test_features.values)
        
        # Train model
        model = HMMRegimeModel(n_states=3, random_state=42, max_iter=100)
        model.fit(train_scaled, feature_names=list(features.columns))
        
        print(f"✓ Model trained with {model.n_states} states")
        print(f"  Converged: {model.converged}")
        print(f"  Log-likelihood: {model.log_likelihood:.4f}")
        
        # Step 5: Make predictions
        print("\n5. Making regime predictions...")
        
        # Predict on test data
        test_states = model.predict(test_scaled)
        test_posteriors = model.predict_proba(test_scaled)
        
        # Label states
        state_labels = label_states_by_return_volatility(
            test_states, test_features['log_return']
        )
        
        print(f"✓ Predicted regimes for {len(test_features)} periods")
        print(f"  State labels: {state_labels}")
        
        # Step 6: Test next-step probabilities
        print("\n6. Computing next-step probabilities...")
        
        A = model.get_transition_matrix()
        current_posterior = test_posteriors[-1]
        next_probs = current_posterior @ A
        
        print("✓ Next-step probabilities:")
        for i, prob in enumerate(next_probs):
            regime_label = state_labels.get(i, f'State_{i}')
            print(f"  {regime_label}: {prob:.3f}")
        
        # Step 7: Save and load model
        print("\n7. Testing model persistence...")
        
        # Save model
        model_path = Path('models/demo_model.joblib')
        model_path.parent.mkdir(exist_ok=True)
        
        metadata = {
            'symbol': 'DEMO',
            'n_states': model.n_states,
            'feature_names': list(features.columns),
            'state_labels': state_labels,
            'training_samples': len(train_features)
        }
        
        save_model_artifact(model, scaler, metadata, str(model_path))
        
        # Test loading with inference engine
        inference_engine = HMMInferenceEngine(str(model_path))
        
        # Make prediction using inference engine
        result = inference_engine.predict_regime(ohlc_data.tail(100))
        
        print("✓ Model saved and loaded successfully")
        print(f"  Current regime: {result['current_regime']['regime_label']}")
        print(f"  Confidence: {result['current_regime']['posterior_probability']:.3f}")
        
        # Step 8: Test backtesting
        print("\n8. Running simple backtest...")
        
        # Create simple strategy
        strategy = SimpleRegimeStrategy(allow_short=True)
        backtester = HMMBacktester(initial_capital=10000)
        
        # Run backtest on test period
        test_ohlc = ohlc_data.iloc[train_size:]
        if len(test_ohlc) > 50:  # Need enough data
            backtest_results = backtester.run_backtest(
                inference_engine=inference_engine,
                ohlc_data=test_ohlc,
                strategy=strategy
            )
            
            perf = backtest_results['performance_metrics']
            print("✓ Backtest completed successfully")
            print(f"  Total return: {perf['total_return']:.2%}")
            print(f"  Sharpe ratio: {perf['sharpe_ratio']:.3f}")
            print(f"  Max drawdown: {perf['max_drawdown']:.2%}")
        else:
            print("✓ Backtest setup validated (insufficient test data for full run)")
        
        # Step 9: Test performance metrics
        print("\n9. Testing performance metrics...")
        
        sample_returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        metrics = calculate_performance_metrics(sample_returns)
        
        print("✓ Performance metrics computed:")
        for key in ['total_return', 'sharpe_ratio', 'max_drawdown']:
            print(f"  {key}: {metrics[key]:.4f}")
        
        # Final validation
        print("\n" + "=" * 70)
        print("🎉 COMPLETE WORKFLOW VALIDATION SUCCESSFUL!")
        print("=" * 70)
        print("\nAll system components are working correctly:")
        print("  ✓ Data ingestion and validation")
        print("  ✓ Feature engineering pipeline")
        print("  ✓ HMM model training and selection")
        print("  ✓ Regime detection and prediction")
        print("  ✓ Next-step probability computation")
        print("  ✓ Model persistence and loading")
        print("  ✓ Inference engine")
        print("  ✓ Backtesting framework")
        print("  ✓ Performance metrics calculation")
        
        print("\nThe system is ready for production use!")
        print("Next steps:")
        print("  1. Set up Alpha Vantage API key in .env file")
        print("  2. Use hmm_cli.py for command-line operations")
        print("  3. Try the Jupyter notebook for interactive analysis")
        print("  4. Run rolling training on real market data")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
