"""
Feature engineering module for HMM market regime detection.

Implements the exact 5 features specified:
1. Log returns
2. Rolling volatility (sample std)
3. RSI (14-period)
4. MACD histogram (MACD - signal line)
5. Bollinger Band Width (normalized)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils import validate_ohlc_data, setup_logging

logger = setup_logging()


def compute_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    vol_window: int = 20,
    macd_short: int = 12,
    macd_long: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_k: float = 2.0
) -> pd.DataFrame:
    """
    Compute the 5 technical features for HMM training.
    
    Args:
        df: DataFrame with OHLC columns and datetime index (sorted ascending)
        rsi_period: RSI lookback period
        vol_window: Rolling volatility window
        macd_short: MACD fast EMA period
        macd_long: MACD slow EMA period
        macd_signal: MACD signal line EMA period
        bb_window: Bollinger Bands window
        bb_k: Bollinger Bands standard deviation multiplier
        
    Returns:
        DataFrame with features: ['log_return', 'volatility', 'RSI', 'MACD', 'BBW']
    """
    logger.info("Computing technical features")
    
    if not validate_ohlc_data(df):
        logger.warning("Input OHLC data failed validation checks")
    
    df = df.copy()
    
    # 1. Log returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Rolling volatility (sample standard deviation)
    df['volatility'] = df['log_return'].rolling(
        window=vol_window, 
        min_periods=vol_window
    ).std()
    
    # 3. RSI (Relative Strength Index)
    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
    
    # 4. MACD Histogram (MACD - Signal Line)
    df['MACD'] = compute_macd_histogram(
        df['Close'], 
        short=macd_short, 
        long=macd_long, 
        signal=macd_signal
    )
    
    # 5. Bollinger Band Width (normalized)
    df['BBW'] = compute_bollinger_band_width(
        df['Close'], 
        window=bb_window, 
        k=bb_k
    )
    
    # Select only feature columns
    feature_cols = ['log_return', 'volatility', 'RSI', 'MACD', 'BBW']
    features = df[feature_cols].copy()
    
    # Remove rows with NaN values (from initial rolling windows)
    initial_length = len(features)
    features = features.dropna()
    final_length = len(features)
    
    logger.info(f"Features computed. Dropped {initial_length - final_length} rows with NaN")
    logger.info(f"Final feature matrix shape: {features.shape}")
    
    # Convert to float64 for numerical stability
    features = features.astype(np.float64)
    
    return features


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    Using simple rolling mean approach for transparency.
    For production, consider Wilder's smoothing method.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Simple rolling mean (alternative: use Wilder's exponential smoothing)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Handle division by zero
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    
    # Fill NaN values with neutral RSI (50)
    rsi = rsi.fillna(50.0)
    
    return rsi


def compute_macd_histogram(
    prices: pd.Series, 
    short: int = 12, 
    long: int = 26, 
    signal: int = 9
) -> pd.Series:
    """
    Compute MACD Histogram (MACD line - Signal line).
    """
    # Exponential moving averages
    ema_short = prices.ewm(span=short, adjust=False).mean()
    ema_long = prices.ewm(span=long, adjust=False).mean()
    
    # MACD line
    macd_line = ema_short - ema_long
    
    # Signal line (EMA of MACD line)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # MACD Histogram
    macd_histogram = macd_line - signal_line
    
    return macd_histogram


def compute_bollinger_band_width(
    prices: pd.Series, 
    window: int = 20, 
    k: float = 2.0
) -> pd.Series:
    """
    Compute normalized Bollinger Band Width.
    
    BBW = (Upper Band - Lower Band) / Moving Average
    """
    # Moving average and standard deviation
    ma = prices.rolling(window=window, min_periods=window).mean()
    std = prices.rolling(window=window, min_periods=window).std()
    
    # Bollinger Bands
    upper_band = ma + k * std
    lower_band = ma - k * std
    
    # Bollinger Band Width (normalized by moving average)
    # Handle potential division by zero
    ma_safe = ma.replace(0, np.nan)
    bbw = (upper_band - lower_band) / ma_safe
    
    # Fill any remaining NaN with median value
    bbw = bbw.fillna(bbw.median())
    
    return bbw


def compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional technical indicators (optional extensions).
    
    These can be used for feature ablation studies or enhanced models.
    """
    df = df.copy()
    
    # Volume-based features (if volume data available)
    if 'Volume' in df.columns:
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Price-based features
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Momentum indicators
    df['roc_5'] = df['Close'].pct_change(periods=5)  # 5-day rate of change
    df['roc_10'] = df['Close'].pct_change(periods=10)  # 10-day rate of change
    
    # Volatility indicators
    df['true_range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            np.abs(df['High'] - df['Close'].shift(1)),
            np.abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['true_range'].rolling(window=14).mean()  # Average True Range
    
    return df


def validate_features(features_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate computed features for anomalies and quality issues.
    """
    required_features = ['log_return', 'volatility', 'RSI', 'MACD', 'BBW']
    
    # Check required columns exist
    missing_cols = [col for col in required_features if col not in features_df.columns]
    if missing_cols:
        return False, f"Missing feature columns: {missing_cols}"
    
    feature_data = features_df[required_features]
    
    # Check for infinite values
    inf_cols = feature_data.columns[np.isinf(feature_data).any()].tolist()
    if inf_cols:
        return False, f"Features contain infinite values: {inf_cols}"
    
    # Check for excessive NaN values
    nan_pct = feature_data.isna().mean()
    high_nan_cols = nan_pct[nan_pct > 0.05].index.tolist()
    if high_nan_cols:
        return False, f"Features have >5% NaN values: {dict(zip(high_nan_cols, nan_pct[high_nan_cols]))}"
    
    # Check reasonable ranges
    validation_issues = []
    
    # RSI should be between 0 and 100
    rsi_out_of_range = ((feature_data['RSI'] < 0) | (feature_data['RSI'] > 100)).sum()
    if rsi_out_of_range > 0:
        validation_issues.append(f"RSI out of range [0,100]: {rsi_out_of_range} values")
    
    # Volatility should be positive
    neg_vol = (feature_data['volatility'] < 0).sum()
    if neg_vol > 0:
        validation_issues.append(f"Negative volatility: {neg_vol} values")
    
    # BBW should be positive
    neg_bbw = (feature_data['BBW'] < 0).sum()
    if neg_bbw > 0:
        validation_issues.append(f"Negative Bollinger Band Width: {neg_bbw} values")
    
    # Check for extreme outliers (beyond 6 standard deviations)
    for col in required_features:
        if col == 'RSI':  # RSI is bounded, skip outlier check
            continue
        mean_val = feature_data[col].mean()
        std_val = feature_data[col].std()
        outliers = ((feature_data[col] - mean_val).abs() > 6 * std_val).sum()
        if outliers > len(feature_data) * 0.01:  # More than 1% outliers
            validation_issues.append(f"{col} has {outliers} extreme outliers (>6σ)")
    
    if validation_issues:
        return False, "; ".join(validation_issues)
    
    return True, "Features are valid"


def get_feature_summary(features_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for computed features.
    """
    required_features = ['log_return', 'volatility', 'RSI', 'MACD', 'BBW']
    feature_data = features_df[required_features]
    
    summary = {
        'n_observations': len(feature_data),
        'date_range': (feature_data.index.min(), feature_data.index.max()),
        'missing_values': feature_data.isna().sum().to_dict(),
        'statistics': feature_data.describe().to_dict()
    }
    
    return summary


def load_and_compute_features(
    data_file: str, 
    **feature_params
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load OHLC data and compute features in one step.
    
    Returns:
        Tuple of (features_df, summary_dict)
    """
    # Load data
    if data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        raise ValueError("Data file must be .csv or .parquet")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Compute features
    features = compute_features(df, **feature_params)
    
    # Validate
    is_valid, validation_msg = validate_features(features)
    if not is_valid:
        logger.warning(f"Feature validation failed: {validation_msg}")
    
    # Generate summary
    summary = get_feature_summary(features)
    summary['validation'] = {'is_valid': is_valid, 'message': validation_msg}
    
    return features, summary


if __name__ == "__main__":
    # Example usage for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute features from OHLC data")
    parser.add_argument('--data-file', required=True, help='Path to OHLC data file')
    parser.add_argument('--output-file', help='Path to save computed features')
    
    args = parser.parse_args()
    
    try:
        features, summary = load_and_compute_features(args.data_file)
        
        print("Feature computation complete:")
        print(f"  Shape: {features.shape}")
        print(f"  Date range: {summary['date_range']}")
        print(f"  Validation: {summary['validation']['message']}")
        
        if args.output_file:
            if args.output_file.endswith('.parquet'):
                features.to_parquet(args.output_file)
            else:
                features.to_csv(args.output_file)
            print(f"  Saved to: {args.output_file}")
            
    except Exception as e:
        logger.error(f"Feature computation failed: {e}")
        raise
