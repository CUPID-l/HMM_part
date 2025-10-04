"""
Production inference API for HMM market regime detection.

Provides real-time regime prediction with next-step probabilities.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils import (
    setup_logging, load_model_artifact, get_latest_model_path, 
    format_currency_pair, ensure_directory_exists
)
from features import compute_features
from model_wrapper import HMMRegimeModel

logger = setup_logging()


class HMMInferenceEngine:
    """
    Production inference engine for HMM regime detection.
    
    Handles model loading, feature computation, scaling, and prediction
    with proper error handling and logging.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize inference engine with a trained model.
        
        Args:
            model_path: Path to saved model artifact
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model artifact
        self.artifact = load_model_artifact(str(self.model_path))
        self.model = self.artifact['model']
        self.scaler = self.artifact['scaler']
        self.metadata = self.artifact['metadata']
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model: {self.metadata['n_states']} states, "
                   f"trained on {self.metadata['start_date']} to {self.metadata['end_date']}")
        
        # Extract model information
        self.n_states = self.metadata['n_states']
        self.feature_names = self.metadata['feature_names']
        self.state_labels = self.metadata['state_labels']
        self.symbol = self.metadata.get('symbol', 'Unknown')
        
        # Get transition matrix for next-step predictions
        self.transition_matrix = self.model.get_transition_matrix()
        
        # Validate model is fitted
        if not self.model.is_fitted:
            raise ValueError("Loaded model is not fitted")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = {
            'model_path': str(self.model_path),
            'symbol': self.symbol,
            'n_states': self.n_states,
            'state_labels': self.state_labels,
            'feature_names': self.feature_names,
            'training_window': {
                'start_date': self.metadata['start_date'],
                'end_date': self.metadata['end_date'],
                'n_samples': self.metadata.get('n_samples', 'Unknown')
            },
            'model_performance': {
                'train_log_likelihood': self.metadata.get('train_log_likelihood'),
                'converged': self.metadata.get('converged')
            },
            'transition_matrix': self.transition_matrix.tolist(),
            'created_at': self.artifact.get('created_at'),
            'version': self.artifact.get('version')
        }
        
        return info
    
    def compute_and_scale_features(self, ohlc_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute features from OHLC data and apply model scaling.
        
        Args:
            ohlc_df: DataFrame with OHLC columns and datetime index
            
        Returns:
            Tuple of (scaled_features, raw_features_df)
        """
        # Compute features using same parameters as training
        features_df = compute_features(ohlc_df)
        
        # Ensure we have the required features in the right order
        if not all(feat in features_df.columns for feat in self.feature_names):
            missing = [f for f in self.feature_names if f not in features_df.columns]
            raise ValueError(f"Missing required features: {missing}")
        
        # Select and order features to match training
        features_ordered = features_df[self.feature_names].copy()
        
        # Remove NaN rows
        features_clean = features_ordered.dropna()
        if len(features_clean) == 0:
            raise ValueError("No valid feature data after removing NaN values")
        
        # Apply scaling using model's fitted scaler
        scaled_features = self.scaler.transform(features_clean.values)
        
        logger.debug(f"Computed and scaled {len(features_clean)} feature observations")
        
        return scaled_features, features_clean
    
    def predict_regime(
        self, 
        ohlc_df: pd.DataFrame,
        return_probabilities: bool = True,
        return_sequence: bool = False
    ) -> Dict[str, Any]:
        """
        Predict market regime from OHLC data.
        
        Args:
            ohlc_df: DataFrame with OHLC data
            return_probabilities: Whether to compute posterior probabilities
            return_sequence: Whether to return full sequence or just latest
            
        Returns:
            Dictionary with prediction results
        """
        # Compute and scale features
        scaled_features, features_df = self.compute_and_scale_features(ohlc_df)
        
        if len(scaled_features) == 0:
            raise ValueError("No valid features computed from input data")
        
        # Viterbi decoding (most likely state sequence)
        viterbi_states = self.model.predict(scaled_features)
        
        # Posterior probabilities (forward-backward)
        posterior_probs = None
        if return_probabilities:
            posterior_probs = self.model.predict_proba(scaled_features)
        
        # Get current (latest) predictions
        current_state = viterbi_states[-1]
        current_posterior = posterior_probs[-1] if posterior_probs is not None else None
        current_date = features_df.index[-1]
        
        # Map state indices to labels
        current_regime = self.state_labels.get(current_state, f"State_{current_state}")
        
        # Compute next-step probabilities
        next_step_probs = self.compute_next_step_probabilities(
            current_posterior=current_posterior,
            current_state=current_state
        )
        
        # Build result dictionary
        result = {
            'timestamp': current_date.isoformat(),
            'current_regime': {
                'state_index': int(current_state),
                'regime_label': current_regime,
                'posterior_probability': float(current_posterior[current_state]) if current_posterior is not None else None
            },
            'next_step_probabilities': next_step_probs,
            'model_info': {
                'symbol': self.symbol,
                'n_states': self.n_states,
                'model_path': str(self.model_path)
            }
        }
        
        # Add sequence data if requested
        if return_sequence:
            regime_sequence = [self.state_labels.get(s, f"State_{s}") for s in viterbi_states]
            result['sequence'] = {
                'dates': [d.isoformat() for d in features_df.index],
                'states': viterbi_states.tolist(),
                'regimes': regime_sequence,
                'posterior_probabilities': posterior_probs.tolist() if posterior_probs is not None else None
            }
        
        # Add additional probabilities if computed
        if return_probabilities and current_posterior is not None:
            result['current_posterior_distribution'] = {
                self.state_labels.get(i, f"State_{i}"): float(prob) 
                for i, prob in enumerate(current_posterior)
            }
        
        return result
    
    def compute_next_step_probabilities(
        self, 
        current_posterior: Optional[np.ndarray] = None,
        current_state: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute next-step regime probabilities using transition matrix.
        
        Args:
            current_posterior: Current posterior distribution over states
            current_state: Current most likely state
            
        Returns:
            Dictionary mapping regime labels to next-step probabilities
        """
        if current_posterior is not None:
            # P(S_{t+1}=j) = sum_i P(S_t=i | obs) * A[i,j]
            next_probs = current_posterior @ self.transition_matrix
        elif current_state is not None:
            # P(S_{t+1}=j) = A[current_state, j]
            next_probs = self.transition_matrix[current_state]
        else:
            # Fallback: uniform distribution
            next_probs = np.ones(self.n_states) / self.n_states
        
        # Map to regime labels
        next_step_probs = {
            self.state_labels.get(i, f"State_{i}"): float(prob)
            for i, prob in enumerate(next_probs)
        }
        
        return next_step_probs
    
    def predict_from_file(self, data_file: str, **kwargs) -> Dict[str, Any]:
        """
        Predict regime from data file.
        
        Args:
            data_file: Path to OHLC data file (CSV or Parquet)
            **kwargs: Additional arguments for predict_regime
            
        Returns:
            Prediction results
        """
        # Load data
        if data_file.endswith('.parquet'):
            ohlc_df = pd.read_parquet(data_file)
        elif data_file.endswith('.csv'):
            ohlc_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        else:
            raise ValueError("Data file must be .csv or .parquet")
        
        # Ensure datetime index
        if not isinstance(ohlc_df.index, pd.DatetimeIndex):
            ohlc_df.index = pd.to_datetime(ohlc_df.index)
        
        # Sort by date
        ohlc_df = ohlc_df.sort_index()
        
        return self.predict_regime(ohlc_df, **kwargs)
    
    def predict_streaming(
        self, 
        new_ohlc_data: Union[Dict, pd.Series, pd.DataFrame],
        historical_data: Optional[pd.DataFrame] = None,
        min_history_length: int = 100
    ) -> Dict[str, Any]:
        """
        Predict regime for streaming data (single new observation).
        
        Args:
            new_ohlc_data: New OHLC observation
            historical_data: Historical OHLC data for context
            min_history_length: Minimum history needed for features
            
        Returns:
            Prediction results
        """
        # Convert new data to DataFrame row
        if isinstance(new_ohlc_data, dict):
            new_row = pd.DataFrame([new_ohlc_data])
            if 'timestamp' in new_ohlc_data:
                new_row.index = pd.to_datetime([new_ohlc_data['timestamp']])
            else:
                new_row.index = pd.to_datetime([datetime.now()])
        elif isinstance(new_ohlc_data, pd.Series):
            new_row = new_ohlc_data.to_frame().T
        elif isinstance(new_ohlc_data, pd.DataFrame):
            new_row = new_ohlc_data.copy()
        else:
            raise ValueError("new_ohlc_data must be dict, Series, or DataFrame")
        
        # Combine with historical data
        if historical_data is not None:
            combined_data = pd.concat([historical_data, new_row]).sort_index()
        else:
            combined_data = new_row
        
        # Check if we have enough data
        if len(combined_data) < min_history_length:
            raise ValueError(f"Insufficient historical data: {len(combined_data)} < {min_history_length}")
        
        # Predict on combined data (features need historical context)
        result = self.predict_regime(combined_data, return_sequence=False)
        
        # Add streaming-specific metadata
        result['streaming_info'] = {
            'new_data_timestamp': new_row.index[-1].isoformat(),
            'historical_data_length': len(historical_data) if historical_data is not None else 0,
            'total_data_length': len(combined_data)
        }
        
        return result
    
    def batch_predict(
        self, 
        data_files: List[str],
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run batch predictions on multiple data files.
        
        Args:
            data_files: List of paths to OHLC data files
            output_file: Optional path to save results
            
        Returns:
            List of prediction results
        """
        results = []
        
        logger.info(f"Running batch prediction on {len(data_files)} files")
        
        for i, data_file in enumerate(data_files):
            try:
                logger.info(f"Processing file {i+1}/{len(data_files)}: {data_file}")
                result = self.predict_from_file(data_file)
                result['source_file'] = data_file
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {data_file}: {e}")
                results.append({
                    'source_file': data_file,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save results if output file specified
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Batch results saved to {output_file}")
        
        return results


def load_inference_engine(
    model_path: Optional[str] = None,
    symbol: Optional[str] = None,
    models_dir: str = "models"
) -> HMMInferenceEngine:
    """
    Load inference engine with automatic model selection.
    
    Args:
        model_path: Explicit path to model file
        symbol: Symbol to find latest model for
        models_dir: Directory containing models
        
    Returns:
        Initialized HMMInferenceEngine
    """
    if model_path:
        return HMMInferenceEngine(model_path)
    elif symbol:
        latest_path = get_latest_model_path(models_dir, symbol)
        if not latest_path:
            raise FileNotFoundError(f"No models found for symbol {symbol} in {models_dir}")
        return HMMInferenceEngine(latest_path)
    else:
        # Find the most recent model in directory
        models_path = Path(models_dir)
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        model_files = list(models_path.glob("*.joblib"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return HMMInferenceEngine(str(latest_model))


def main():
    """Command line interface for inference."""
    parser = argparse.ArgumentParser(description="HMM regime inference")
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model-path', help='Path to specific model file')
    model_group.add_argument('--symbol', help='Symbol to find latest model for')
    
    # Input data
    parser.add_argument('--data-file', help='OHLC data file for prediction')
    parser.add_argument('--models-dir', default='models', help='Models directory')
    
    # Output options
    parser.add_argument('--output-file', help='Save results to file')
    parser.add_argument('--return-sequence', action='store_true', 
                       help='Return full state sequence')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    try:
        # Load inference engine
        if args.model_path:
            engine = HMMInferenceEngine(args.model_path)
        else:
            engine = load_inference_engine(symbol=args.symbol, models_dir=args.models_dir)
        
        # Print model info
        model_info = engine.get_model_info()
        print(f"Loaded model: {model_info['symbol']} ({model_info['n_states']} states)")
        print(f"Training period: {model_info['training_window']['start_date']} to {model_info['training_window']['end_date']}")
        
        # Run prediction if data file provided
        if args.data_file:
            result = engine.predict_from_file(
                args.data_file, 
                return_sequence=args.return_sequence
            )
            
            # Print current regime
            current = result['current_regime']
            print(f"\nCurrent regime: {current['regime_label']} (confidence: {current['posterior_probability']:.3f})")
            
            # Print next-step probabilities
            print("\nNext-step probabilities:")
            for regime, prob in result['next_step_probabilities'].items():
                print(f"  {regime}: {prob:.3f}")
            
            # Save results if requested
            if args.output_file:
                if args.format == 'json':
                    import json
                    with open(args.output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                elif args.format == 'csv':
                    # Convert to DataFrame for CSV output
                    if 'sequence' in result:
                        seq_df = pd.DataFrame({
                            'date': result['sequence']['dates'],
                            'state': result['sequence']['states'],
                            'regime': result['sequence']['regimes']
                        })
                        seq_df.to_csv(args.output_file, index=False)
                    else:
                        # Single prediction
                        pred_df = pd.DataFrame([{
                            'timestamp': result['timestamp'],
                            'regime': current['regime_label'],
                            'confidence': current['posterior_probability']
                        }])
                        pred_df.to_csv(args.output_file, index=False)
                
                print(f"\nResults saved to {args.output_file}")
        else:
            print("\nModel loaded successfully. Use --data-file to make predictions.")
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
