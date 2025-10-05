"""
HMM Prediction Validation

This module validates HMM predictions against historical data without simulating trades.
Focus: Prove the HMM generates accurate predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, Any, Tuple
from model_wrapper import HMMRegimeModel
from features import compute_features

logger = logging.getLogger(__name__)


class HMMPredictionValidator:
    """Validates HMM predictions against historical data."""
    
    def __init__(self, model_path: str, data_path: str):
        """
        Initialize validator.
        
        Args:
            model_path: Path to trained HMM model
            data_path: Path to historical OHLC data
        """
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.scaler = None
        self.state_labels = None
        self.feature_names = None
        
    def load_model(self) -> None:
        """Load trained HMM model."""
        import joblib
        
        artifact = joblib.load(self.model_path)
        self.model = artifact['model']
        self.scaler = artifact['scaler']
        self.state_labels = artifact.get('state_labels', {})
        self.feature_names = artifact.get('feature_names', [])
        
        logger.info(f"Loaded model from {self.model_path}")
        logger.info(f"Model: {self.model.n_states} states")
        logger.info(f"State labels: {self.state_labels}")
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load historical data and compute features.
        
        Returns:
            ohlc_data: Original OHLC dataframe
            X: Scaled feature matrix
        """
        # Load OHLC data
        ohlc_data = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(ohlc_data)} OHLC records")
        logger.info(f"Date range: {ohlc_data.index[0]} to {ohlc_data.index[-1]}")
        
        # Compute features
        features = compute_features(ohlc_data)
        logger.info(f"Computed features: {list(features.columns)}")
        
        # Scale features
        X = self.scaler.transform(features)
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Align indices
        ohlc_data = ohlc_data.loc[features.index]
        
        return ohlc_data, X
        
    def split_data(self, X: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            train_ratio: Ratio of training data
            
        Returns:
            X_train, X_test
        """
        split_idx = int(len(X) * train_ratio)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        
        logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test
        
    def predict_states(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict hidden states and posterior probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            states: Predicted state sequence
            posteriors: Posterior probabilities for each state
        """
        states = self.model.predict(X)
        posteriors = self.model.predict_proba(X)
        
        logger.info(f"Generated {len(states)} state predictions")
        
        return states, posteriors
        
    def calculate_state_statistics(self, states: np.ndarray, posteriors: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics about predicted states.
        
        Args:
            states: Predicted state sequence
            posteriors: Posterior probabilities
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # State distribution
        unique_states, counts = np.unique(states, return_counts=True)
        state_dist = {
            self.state_labels.get(s, f"State_{s}"): int(count)
            for s, count in zip(unique_states, counts)
        }
        stats['state_distribution'] = state_dist
        stats['state_frequencies'] = {
            label: count / len(states)
            for label, count in state_dist.items()
        }
        
        # Average posterior confidence
        max_posteriors = posteriors.max(axis=1)
        stats['mean_confidence'] = float(max_posteriors.mean())
        stats['median_confidence'] = float(np.median(max_posteriors))
        stats['min_confidence'] = float(max_posteriors.min())
        stats['max_confidence'] = float(max_posteriors.max())
        
        # State transitions
        transitions = np.sum(states[1:] != states[:-1])
        stats['total_transitions'] = int(transitions)
        stats['transition_rate'] = float(transitions / len(states))
        
        # Average duration in each state
        durations = {}
        for state_idx, label in self.state_labels.items():
            state_mask = (states == state_idx)
            if state_mask.any():
                # Calculate run lengths
                runs = np.diff(np.where(np.concatenate(([state_mask[0]], 
                                                        state_mask[:-1] != state_mask[1:], 
                                                        [True])))[0])[::2]
                durations[label] = {
                    'mean_duration': float(runs.mean()) if len(runs) > 0 else 0,
                    'max_duration': int(runs.max()) if len(runs) > 0 else 0,
                    'min_duration': int(runs.min()) if len(runs) > 0 else 0
                }
        
        stats['state_durations'] = durations
        
        return stats
        
    def validate_regime_predictions(self, ohlc_data: pd.DataFrame, states: np.ndarray, 
                                   posteriors: np.ndarray) -> Dict[str, Any]:
        """
        Validate regime predictions against actual price movements.
        
        Args:
            ohlc_data: OHLC dataframe
            states: Predicted state sequence
            posteriors: Posterior probabilities
            
        Returns:
            Validation metrics
        """
        results = {}
        
        # Calculate returns for each regime
        # Handle both 'close' and 'Close' column names
        close_col = 'Close' if 'Close' in ohlc_data.columns else 'close'
        returns = ohlc_data[close_col].pct_change().values[1:]  # Skip first NaN
        states_aligned = states[1:]  # Align with returns
        
        regime_performance = {}
        for state_idx, label in self.state_labels.items():
            mask = (states_aligned == state_idx)
            if mask.any():
                regime_returns = returns[mask]
                regime_performance[label] = {
                    'mean_return': float(regime_returns.mean()),
                    'std_return': float(regime_returns.std()),
                    'median_return': float(np.median(regime_returns)),
                    'positive_days': int((regime_returns > 0).sum()),
                    'negative_days': int((regime_returns < 0).sum()),
                    'win_rate': float((regime_returns > 0).mean()),
                    'total_days': int(mask.sum())
                }
        
        results['regime_performance'] = regime_performance
        
        # Regime consistency check
        # Bull should have positive returns, Bear negative, Sideways near-zero
        consistency_score = 0
        total_checks = 0
        
        for label, perf in regime_performance.items():
            if label == 'Bull' and perf['mean_return'] > 0:
                consistency_score += 1
            elif label == 'Bear' and perf['mean_return'] < 0:
                consistency_score += 1
            elif label == 'Sideways' and abs(perf['mean_return']) < 0.0001:
                consistency_score += 1
            total_checks += 1
        
        results['regime_consistency'] = float(consistency_score / total_checks) if total_checks > 0 else 0
        
        # Next-day direction prediction accuracy
        # Check if regime predicts next day's direction
        next_returns = returns[1:]
        current_states = states_aligned[:-1]
        
        direction_accuracy = {}
        for state_idx, label in self.state_labels.items():
            mask = (current_states == state_idx)
            if mask.any():
                state_next_returns = next_returns[mask]
                
                # Predict direction based on regime
                if label == 'Bull':
                    predicted_positive = True
                elif label == 'Bear':
                    predicted_positive = False
                else:  # Sideways
                    predicted_positive = None
                
                if predicted_positive is not None:
                    actual_positive = (state_next_returns > 0)
                    if predicted_positive:
                        accuracy = actual_positive.mean()
                    else:
                        accuracy = (~actual_positive).mean()
                    
                    direction_accuracy[label] = float(accuracy)
        
        results['direction_accuracy'] = direction_accuracy
        
        # Overall metrics
        results['total_samples'] = int(len(states))
        results['date_range'] = {
            'start': str(ohlc_data.index[0]),
            'end': str(ohlc_data.index[-1])
        }
        
        return results
        
    def run_validation(self, output_file: str = None) -> Dict[str, Any]:
        """
        Run complete validation pipeline.
        
        Args:
            output_file: Optional path to save results
            
        Returns:
            Validation results dictionary
        """
        logger.info("="*70)
        logger.info("HMM PREDICTION VALIDATION")
        logger.info("="*70)
        
        # Load model
        self.load_model()
        
        # Load and prepare data
        ohlc_data, X = self.load_and_prepare_data()
        
        # Generate predictions
        logger.info("\nGenerating predictions...")
        states, posteriors = self.predict_states(X)
        
        # Calculate statistics
        logger.info("\nCalculating statistics...")
        state_stats = self.calculate_state_statistics(states, posteriors)
        
        # Validate predictions
        logger.info("\nValidating predictions...")
        validation_results = self.validate_regime_predictions(ohlc_data, states, posteriors)
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'data_path': str(self.data_path),
            'model_info': {
                'n_states': int(self.model.n_states),
                'state_labels': self.state_labels,
                'feature_names': self.feature_names
            },
            'state_statistics': state_stats,
            'validation_results': validation_results
        }
        
        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nResults saved to {output_file}")
        
        # Print summary
        self._print_summary(results)
        
        return results
        
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print validation summary."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        
        # State distribution
        logger.info("\n📊 State Distribution:")
        for label, freq in results['state_statistics']['state_frequencies'].items():
            count = results['state_statistics']['state_distribution'][label]
            logger.info(f"  {label:10s}: {freq:6.2%} ({count:4d} days)")
        
        # Confidence
        stats = results['state_statistics']
        logger.info(f"\n🎯 Prediction Confidence:")
        logger.info(f"  Mean:   {stats['mean_confidence']:.2%}")
        logger.info(f"  Median: {stats['median_confidence']:.2%}")
        logger.info(f"  Range:  {stats['min_confidence']:.2%} - {stats['max_confidence']:.2%}")
        
        # Regime consistency
        consistency = results['validation_results']['regime_consistency']
        logger.info(f"\n✓ Regime Consistency: {consistency:.2%}")
        logger.info("  (Do regimes match their expected behavior?)")
        
        # Direction accuracy
        logger.info(f"\n📈 Next-Day Direction Prediction:")
        for label, accuracy in results['validation_results']['direction_accuracy'].items():
            logger.info(f"  {label:10s}: {accuracy:6.2%} accuracy")
        
        # Regime performance
        logger.info(f"\n💰 Regime Performance (Mean Daily Return):")
        for label, perf in results['validation_results']['regime_performance'].items():
            mean_ret = perf['mean_return'] * 100
            win_rate = perf['win_rate']
            logger.info(f"  {label:10s}: {mean_ret:+.4f}% (Win rate: {win_rate:.2%})")
        
        logger.info("\n" + "="*70)
        logger.info("✅ VALIDATION COMPLETE")
        logger.info("="*70)


def main():
    """Command-line interface for validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate HMM predictions")
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-file', required=True, help='Path to OHLC data')
    parser.add_argument('--output-file', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run validation
    validator = HMMPredictionValidator(args.model_path, args.data_file)
    results = validator.run_validation(args.output_file)
    
    return results


if __name__ == "__main__":
    main()
