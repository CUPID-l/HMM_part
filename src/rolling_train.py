"""
Rolling window HMM training pipeline.

Implements walk-forward training with proper model persistence and validation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import argparse
from sklearn.preprocessing import StandardScaler
import joblib

from utils import (
    setup_logging, set_random_seeds, label_states_by_return_volatility,
    save_model_artifact, ensure_directory_exists, format_currency_pair,
    validate_feature_data, calculate_performance_metrics
)
from features import load_and_compute_features, compute_features
from model_wrapper import HMMRegimeModel, select_best_model
from ingest import AlphaVantageIngester

logger = setup_logging()


class RollingHMMTrainer:
    """Rolling window HMM training with walk-forward validation."""
    
    def __init__(
        self,
        window_size: int = 252,  # 1 year of daily data
        step_size: int = 21,     # Monthly steps
        n_states: int = 3,
        min_train_size: int = 100,
        validation_size: int = 21,
        models_dir: str = "models",
        **model_params
    ):
        """
        Initialize rolling trainer.
        
        Args:
            window_size: Training window size in periods
            step_size: Step size for rolling window
            n_states: Number of HMM states
            min_train_size: Minimum training data size
            validation_size: Validation window size
            models_dir: Directory to save models
            **model_params: Additional HMM model parameters
        """
        self.window_size = window_size
        self.step_size = step_size
        self.n_states = n_states
        self.min_train_size = min_train_size
        self.validation_size = validation_size
        self.models_dir = Path(models_dir)
        self.model_params = model_params
        
        # Training history
        self.training_history = []
        self.model_artifacts = []
        
        ensure_directory_exists(str(self.models_dir / "dummy"))
    
    def prepare_training_data(
        self, 
        features_df: pd.DataFrame,
        start_idx: int,
        window_size: int
    ) -> Tuple[pd.DataFrame, StandardScaler, Dict]:
        """
        Prepare training data with scaling and validation.
        
        Returns:
            Tuple of (scaled_features, scaler, metadata)
        """
        # Extract training window
        end_idx = start_idx + window_size
        train_data = features_df.iloc[start_idx:end_idx].copy()
        
        # Validate data quality
        is_valid, validation_msg = validate_feature_data(train_data)
        if not is_valid:
            raise ValueError(f"Training data validation failed: {validation_msg}")
        
        # Remove any remaining NaN values
        train_data_clean = train_data.dropna()
        if len(train_data_clean) < self.min_train_size:
            raise ValueError(f"Insufficient clean training data: {len(train_data_clean)} < {self.min_train_size}")
        
        # Fit scaler on training data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(train_data_clean.values)
        
        # Prepare metadata
        metadata = {
            'start_date': train_data.index[0],
            'end_date': train_data.index[-1],
            'n_samples': len(train_data_clean),
            'n_features': train_data_clean.shape[1],
            'feature_names': list(train_data_clean.columns),
            'validation_message': validation_msg,
            'data_summary': train_data_clean.describe().to_dict()
        }
        
        return train_data_clean, scaler, scaled_features, metadata
    
    def fit_and_evaluate_model(
        self,
        scaled_features: np.ndarray,
        train_data: pd.DataFrame,
        scaler: StandardScaler,
        validation_data: Optional[pd.DataFrame] = None,
        random_state: int = 42
    ) -> Tuple[HMMRegimeModel, Dict]:
        """
        Fit HMM model and evaluate performance.
        
        Returns:
            Tuple of (fitted_model, evaluation_metrics)
        """
        # Set random seed for reproducibility
        set_random_seeds(random_state)
        
        # Initialize and fit model
        model = HMMRegimeModel(
            n_states=self.n_states,
            random_state=random_state,
            **self.model_params
        )
        
        logger.info(f"Fitting HMM with {self.n_states} states on {len(scaled_features)} samples")
        model.fit(scaled_features, feature_names=list(train_data.columns))
        
        # Decode training states
        train_states = model.predict(scaled_features)
        train_posteriors = model.predict_proba(scaled_features)
        
        # Label states based on return characteristics
        state_labels = label_states_by_return_volatility(
            train_states, 
            train_data['log_return']
        )
        
        # Compute in-sample metrics
        train_log_likelihood = model.score(scaled_features)
        
        # Calculate regime-wise statistics
        regime_stats = {}
        for state_idx, label in state_labels.items():
            mask = train_states == state_idx
            if mask.sum() > 0:
                regime_returns = train_data['log_return'][mask]
                regime_stats[label] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'frequency': mask.sum() / len(train_states),
                    'avg_duration': self._calculate_avg_duration(train_states, state_idx)
                }
        
        # Validation metrics (if validation data provided)
        val_metrics = {}
        if validation_data is not None:
            try:
                val_features_clean = validation_data.dropna()
                if len(val_features_clean) > 0:
                    val_scaled = scaler.transform(val_features_clean.values)
                    val_log_likelihood = model.score(val_scaled)
                    val_states = model.predict(val_scaled)
                    
                    val_metrics = {
                        'validation_log_likelihood': val_log_likelihood,
                        'validation_n_samples': len(val_features_clean),
                        'validation_state_distribution': np.bincount(val_states, minlength=self.n_states).tolist()
                    }
            except Exception as e:
                logger.warning(f"Validation evaluation failed: {e}")
                val_metrics = {'validation_error': str(e)}
        
        # Compile evaluation metrics
        evaluation = {
            'train_log_likelihood': train_log_likelihood,
            'converged': model.converged,
            'n_iter': getattr(model, 'n_iter', None),
            'state_labels': state_labels,
            'regime_statistics': regime_stats,
            'state_distribution': np.bincount(train_states, minlength=self.n_states).tolist(),
            'transition_matrix': model.get_transition_matrix().tolist(),
            **val_metrics
        }
        
        return model, evaluation
    
    def _calculate_avg_duration(self, states: np.ndarray, target_state: int) -> float:
        """Calculate average duration in a specific state."""
        if len(states) == 0:
            return 0.0
        
        durations = []
        current_duration = 0
        
        for state in states:
            if state == target_state:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        # Don't forget the last duration if it ends with target state
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0
    
    def save_model_artifact_with_metadata(
        self,
        model: HMMRegimeModel,
        scaler: StandardScaler,
        train_metadata: Dict,
        evaluation: Dict,
        symbol: str,
        window_end_date: pd.Timestamp
    ) -> str:
        """Save complete model artifact with all metadata."""
        
        # Extract state labels from evaluation
        state_labels = evaluation.get('state_labels', {})
        feature_names = train_metadata.get('feature_names', ['log_return', 'volatility', 'RSI', 'MACD', 'BBW'])
        
        # Create comprehensive metadata
        artifact_metadata = {
            # Training configuration
            'symbol': symbol,
            'n_states': self.n_states,
            'window_size': self.window_size,
            'step_size': self.step_size,
            
            # Training data info
            **train_metadata,
            
            # Model evaluation
            **evaluation,
            
            # Model parameters
            'model_params': self.model_params,
            
            # Versioning
            'training_timestamp': datetime.now().isoformat(),
            'model_version': '1.0',
        }
        
        # Generate filename
        symbol_formatted = format_currency_pair(symbol)
        date_str = window_end_date.strftime('%Y%m%d')
        filename = f"hmm_{symbol_formatted}_{self.n_states}_win{self.window_size}_end{date_str}.joblib"
        filepath = self.models_dir / filename
        
        # ✅ Save complete artifact with state_labels at top level
        artifact = {
            'model': model,
            'scaler': scaler,
            'state_labels': state_labels,      # ✅ Now at top level!
            'regime_labels': state_labels,      # ✅ Alias for backwards compatibility
            'feature_names': feature_names,     # ✅ Feature names at top level
            'metadata': artifact_metadata,
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        joblib.dump(artifact, filepath)
        
        logger.info(f"Saved model artifact: {filepath}")
        logger.info(f"State labels: {state_labels}")
        return str(filepath)
    
    def train_rolling_windows(
        self,
        features_df: pd.DataFrame,
        symbol: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Run rolling window training across the entire dataset.
        
        Args:
            features_df: DataFrame with computed features
            symbol: Trading symbol for naming
            start_date: Optional start date for training
            end_date: Optional end date for training
            
        Returns:
            Dictionary with training summary and results
        """
        logger.info(f"Starting rolling HMM training for {symbol}")
        logger.info(f"Window size: {self.window_size}, Step size: {self.step_size}")
        
        # Filter data by date range if specified
        if start_date:
            features_df = features_df[features_df.index >= start_date]
        if end_date:
            features_df = features_df[features_df.index <= end_date]
        
        logger.info(f"Training data: {len(features_df)} samples from {features_df.index.min()} to {features_df.index.max()}")
        
        # Reset training history
        self.training_history = []
        self.model_artifacts = []
        
        # Rolling window loop
        n_windows = 0
        successful_windows = 0
        
        for start_idx in range(0, len(features_df) - self.window_size, self.step_size):
            try:
                # Check if we have enough data for window + validation
                remaining_data = len(features_df) - start_idx
                if remaining_data < self.window_size + self.validation_size:
                    logger.info(f"Insufficient data for window + validation at index {start_idx}, stopping")
                    break
                
                n_windows += 1
                window_start_date = features_df.index[start_idx]
                window_end_date = features_df.index[start_idx + self.window_size - 1]
                
                logger.info(f"Training window {n_windows}: {window_start_date.date()} to {window_end_date.date()}")
                
                # Prepare training data
                train_data, scaler, scaled_features, train_metadata = self.prepare_training_data(
                    features_df, start_idx, self.window_size
                )
                
                # Prepare validation data
                val_start_idx = start_idx + self.window_size
                val_end_idx = val_start_idx + self.validation_size
                validation_data = features_df.iloc[val_start_idx:val_end_idx] if val_end_idx <= len(features_df) else None
                
                # Fit and evaluate model
                model, evaluation = self.fit_and_evaluate_model(
                    scaled_features=scaled_features,
                    train_data=train_data,
                    scaler=scaler,
                    validation_data=validation_data,
                    random_state=42 + n_windows  # Different seed per window
                )
                
                # Save model artifact
                artifact_path = self.save_model_artifact_with_metadata(
                    model=model,
                    scaler=scaler,
                    train_metadata=train_metadata,
                    evaluation=evaluation,
                    symbol=symbol,
                    window_end_date=window_end_date
                )
                
                # Record training history
                window_record = {
                    'window_id': n_windows,
                    'start_date': window_start_date,
                    'end_date': window_end_date,
                    'start_idx': start_idx,
                    'n_train_samples': len(train_data),
                    'artifact_path': artifact_path,
                    **evaluation
                }
                
                self.training_history.append(window_record)
                self.model_artifacts.append(artifact_path)
                successful_windows += 1
                
                logger.info(f"Window {n_windows} complete. Train LL: {evaluation['train_log_likelihood']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train window {n_windows} starting at {start_idx}: {e}")
                continue
        
        # Generate training summary
        training_summary = self.generate_training_summary(symbol)
        
        logger.info(f"Rolling training complete: {successful_windows}/{n_windows} windows successful")
        return training_summary
    
    def generate_training_summary(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        if not self.training_history:
            return {'error': 'No successful training windows'}
        
        # Convert to DataFrame for analysis
        history_df = pd.DataFrame(self.training_history)
        
        # Overall statistics
        summary = {
            'symbol': symbol,
            'n_windows': len(self.training_history),
            'date_range': (
                history_df['start_date'].min(),
                history_df['end_date'].max()
            ),
            'avg_train_samples': history_df['n_train_samples'].mean(),
            'window_config': {
                'window_size': self.window_size,
                'step_size': self.step_size,
                'n_states': self.n_states
            },
            
            # Model performance statistics
            'performance_stats': {
                'avg_train_ll': history_df['train_log_likelihood'].mean(),
                'std_train_ll': history_df['train_log_likelihood'].std(),
                'min_train_ll': history_df['train_log_likelihood'].min(),
                'max_train_ll': history_df['train_log_likelihood'].max(),
                'convergence_rate': history_df['converged'].mean() if 'converged' in history_df else None
            },
            
            # Validation statistics (if available)
            'validation_stats': {},
            
            # Model artifacts
            'model_artifacts': self.model_artifacts,
            'latest_model': self.model_artifacts[-1] if self.model_artifacts else None,
            
            # Full training history
            'training_history': self.training_history
        }
        
        # Add validation stats if available
        if 'validation_log_likelihood' in history_df.columns:
            val_ll = history_df['validation_log_likelihood'].dropna()
            if len(val_ll) > 0:
                summary['validation_stats'] = {
                    'avg_val_ll': val_ll.mean(),
                    'std_val_ll': val_ll.std(),
                    'min_val_ll': val_ll.min(),
                    'max_val_ll': val_ll.max()
                }
        
        return summary


def train_symbol(
    symbol: str,
    data_source: str = "file",
    data_path: Optional[str] = None,
    output_dir: str = "models",
    window_size: int = 252,
    step_size: int = 21,
    n_states: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Train HMM models for a single symbol.
    
    Args:
        symbol: Trading symbol
        data_source: 'file' or 'api'
        data_path: Path to data file (if data_source='file')
        output_dir: Directory to save models
        window_size: Training window size
        step_size: Rolling step size
        n_states: Number of HMM states
        **kwargs: Additional parameters
        
    Returns:
        Training summary dictionary
    """
    logger.info(f"Training HMM models for {symbol}")
    
    # Load or fetch data
    if data_source == "file":
        if not data_path:
            raise ValueError("data_path must be provided when data_source='file'")
        features_df, _ = load_and_compute_features(data_path)
    elif data_source == "api":
        # Fetch from API and compute features
        ingester = AlphaVantageIngester()
        raw_df = ingester.fetch_and_save(symbol)
        features_df = compute_features(raw_df)
    else:
        raise ValueError("data_source must be 'file' or 'api'")
    
    # Initialize trainer
    trainer = RollingHMMTrainer(
        window_size=window_size,
        step_size=step_size,
        n_states=n_states,
        models_dir=output_dir,
        **kwargs
    )
    
    # Run rolling training
    results = trainer.train_rolling_windows(features_df, symbol)
    
    return results


def main():
    """Command line interface for rolling training."""
    parser = argparse.ArgumentParser(description="Rolling HMM training pipeline")
    
    # Data arguments
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., EUR/USD)')
    parser.add_argument('--data-source', choices=['file', 'api'], default='file', 
                       help='Data source: file or api')
    parser.add_argument('--data-path', help='Path to data file (required if data-source=file)')
    
    # Training parameters
    parser.add_argument('--window-size', type=int, default=252, 
                       help='Training window size (default: 252 for ~1 year)')
    parser.add_argument('--step-size', type=int, default=21, 
                       help='Rolling step size (default: 21 for ~monthly)')
    parser.add_argument('--n-states', type=int, default=3, 
                       help='Number of HMM states')
    parser.add_argument('--output-dir', default='models', 
                       help='Output directory for models')
    
    # Model parameters
    parser.add_argument('--max-iter', type=int, default=200, 
                       help='Maximum EM iterations')
    parser.add_argument('--covariance-type', default='full', 
                       choices=['full', 'diag', 'tied'],
                       help='Covariance matrix type')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Prepare model parameters
    model_params = {
        'max_iter': args.max_iter,
        'covariance_type': args.covariance_type
    }
    
    try:
        # Run training
        results = train_symbol(
            symbol=args.symbol,
            data_source=args.data_source,
            data_path=args.data_path,
            output_dir=args.output_dir,
            window_size=args.window_size,
            step_size=args.step_size,
            n_states=args.n_states,
            **model_params
        )
        
        # Print results
        print(f"\nTraining completed for {args.symbol}")
        print(f"Windows trained: {results['n_windows']}")
        print(f"Date range: {results['date_range'][0].date()} to {results['date_range'][1].date()}")
        print(f"Average log-likelihood: {results['performance_stats']['avg_train_ll']:.4f}")
        print(f"Latest model: {results['latest_model']}")
        
        if 'validation_stats' in results and results['validation_stats']:
            print(f"Average validation LL: {results['validation_stats']['avg_val_ll']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
