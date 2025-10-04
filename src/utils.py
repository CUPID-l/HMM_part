"""
Utility functions for the HMM market regime detection system.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import joblib


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or 'hmm_system.log')
        ]
    )
    return logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def validate_ohlc_data(df: pd.DataFrame) -> bool:
    """Validate OHLC dataframe structure."""
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        return False
    
    # Check for valid OHLC relationships
    valid_high = (df['High'] >= df[['Open', 'Close']].max(axis=1)).all()
    valid_low = (df['Low'] <= df[['Open', 'Close']].min(axis=1)).all()
    
    return valid_high and valid_low


def clean_ohlc_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare OHLC data."""
    df = df.copy()
    
    # Remove duplicates and sort by date
    df = df.drop_duplicates().sort_index()
    
    # Forward fill missing values (small gaps only)
    df = df.fillna(method='ffill', limit=3)
    
    # Drop remaining NaN rows
    df = df.dropna()
    
    # Remove zero or negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    df = df[df[price_cols].gt(0).all(axis=1)]
    
    return df


def label_states_by_return_volatility(states: np.ndarray, returns: pd.Series) -> Dict[int, str]:
    """
    Label HMM states based on mean return and volatility characteristics.
    
    Args:
        states: Array of decoded states
        returns: Corresponding log returns
        
    Returns:
        Dictionary mapping state index to regime label
    """
    n_states = len(np.unique(states))
    state_stats = []
    
    for state in range(n_states):
        mask = states == state
        if mask.sum() > 0:
            mean_ret = returns[mask].mean()
            vol = returns[mask].std()
            state_stats.append((state, mean_ret, vol))
    
    # Sort by mean return
    state_stats.sort(key=lambda x: x[1])
    
    # Label based on return ranking
    labels = {}
    if len(state_stats) == 2:
        labels[state_stats[0][0]] = 'Bear'
        labels[state_stats[1][0]] = 'Bull'
    elif len(state_stats) >= 3:
        labels[state_stats[0][0]] = 'Bear'
        labels[state_stats[-1][0]] = 'Bull'
        # Middle states are sideways
        for i in range(1, len(state_stats) - 1):
            labels[state_stats[i][0]] = 'Sideways'
    else:
        # Fallback for single state
        labels[state_stats[0][0]] = 'Neutral'
    
    return labels


def save_model_artifact(
    model, 
    scaler, 
    metadata: Dict[str, Any], 
    filepath: str
) -> None:
    """Save complete model artifact with metadata."""
    artifact = {
        'model': model,
        'scaler': scaler,
        'metadata': metadata,
        'created_at': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    joblib.dump(artifact, filepath)


def load_model_artifact(filepath: str) -> Dict[str, Any]:
    """Load complete model artifact."""
    return joblib.load(filepath)


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate standard performance metrics for returns series."""
    if len(returns) == 0:
        return {}
    
    # Annualized metrics (assuming daily returns)
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    n_years = n_periods / 252  # Assuming 252 trading days per year
    
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.02) / annual_vol if annual_vol > 0 else 0  # Assuming 2% risk-free rate
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'hit_rate': (returns > 0).mean(),
        'n_trades': len(returns)
    }


def format_currency_pair(symbol: str) -> str:
    """Format currency pair for file naming."""
    return symbol.replace('/', '_').replace('-', '_').upper()


def get_latest_model_path(models_dir: str, symbol: str, n_states: int = 3) -> Optional[str]:
    """Get the path to the most recent model for a symbol."""
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None
    
    symbol_formatted = format_currency_pair(symbol)
    pattern = f"hmm_{symbol_formatted}_{n_states}_win*_end*.joblib"
    
    matching_files = list(models_dir.glob(pattern))
    if not matching_files:
        return None
    
    # Sort by modification time and return the latest
    latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def validate_feature_data(features_df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate computed features for training."""
    required_features = ['log_return', 'volatility', 'RSI', 'MACD', 'BBW']
    
    # Check required columns
    missing_cols = [col for col in required_features if col not in features_df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for infinite values
    if np.isinf(features_df[required_features]).any().any():
        return False, "Features contain infinite values"
    
    # Check for too many NaN values
    nan_pct = features_df[required_features].isna().mean()
    if (nan_pct > 0.1).any():
        return False, f"Too many NaN values in features: {nan_pct[nan_pct > 0.1].to_dict()}"
    
    # Check minimum data points
    if len(features_df) < 100:
        return False, f"Insufficient data points: {len(features_df)} < 100"
    
    return True, "Features are valid"


def ensure_directory_exists(filepath: str) -> None:
    """Ensure directory exists for given filepath."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
