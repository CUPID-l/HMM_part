"""
Unit tests for feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import compute_features, load_and_compute_features


class TestComputeFeatures:
    """Test suite for compute_features function"""
    
    @pytest.fixture
    def sample_ohlc(self):
        """Create sample OHLC data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        returns = np.random.randn(100) * 0.01
        close_prices = 100 * np.exp(returns.cumsum())
        
        df = pd.DataFrame({
            'Open': close_prices * (1 + np.random.randn(100) * 0.001),
            'High': close_prices * (1 + np.abs(np.random.randn(100) * 0.002)),
            'Low': close_prices * (1 - np.abs(np.random.randn(100) * 0.002)),
            'Close': close_prices,
            'Volume': np.random.randint(10000, 100000, 100)
        }, index=dates)
        
        # Ensure OHLC consistency
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        return df
    
    @pytest.fixture
    def flat_prices(self):
        """Create flat price data (zero volatility)"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Open': [100.0] * 50,
            'High': [100.0] * 50,
            'Low': [100.0] * 50,
            'Close': [100.0] * 50,
        }, index=dates)
        return df
    
    def test_basic_feature_computation(self, sample_ohlc):
        """Test that features are computed correctly on normal data"""
        features = compute_features(sample_ohlc)
        
        # Check shape
        assert features.shape[1] == 5, "Should have exactly 5 features"
        assert len(features) > 0, "Should have some rows after processing"
        assert len(features) < len(sample_ohlc), "Should drop some rows due to indicators"
        
        # Check column names
        expected_cols = ['log_return', 'volatility', 'RSI', 'MACD', 'BBW']
        assert list(features.columns) == expected_cols, f"Expected columns {expected_cols}"
        
        # Check no NaN values after processing
        assert not features.isnull().any().any(), "Should not have NaN values after dropna"
    
    def test_rsi_range(self, sample_ohlc):
        """Test that RSI is always between 0 and 100"""
        features = compute_features(sample_ohlc)
        
        rsi = features['RSI']
        assert rsi.min() >= 0, "RSI minimum should be >= 0"
        assert rsi.max() <= 100, "RSI maximum should be <= 100"
        assert rsi.mean() > 0, "RSI mean should be positive"
        assert rsi.mean() < 100, "RSI mean should be less than 100"
    
    def test_volatility_non_negative(self, sample_ohlc):
        """Test that volatility is always non-negative"""
        features = compute_features(sample_ohlc)
        
        volatility = features['volatility']
        assert (volatility >= 0).all(), "Volatility should always be non-negative"
    
    def test_bbw_non_negative(self, sample_ohlc):
        """Test that Bollinger Band Width is non-negative"""
        features = compute_features(sample_ohlc)
        
        bbw = features['BBW']
        assert (bbw >= 0).all(), "BBW should always be non-negative"
    
    def test_flat_prices(self, flat_prices):
        """Test feature computation on flat prices (zero volatility)"""
        features = compute_features(flat_prices)
        
        # Should still compute features without errors
        assert len(features) > 0, "Should return features even for flat prices"
        
        # Volatility should be zero or near-zero
        assert features['volatility'].max() < 1e-6, "Volatility should be near-zero for flat prices"
        
        # Log returns should be zero
        assert np.abs(features['log_return']).max() < 1e-10, "Returns should be zero for flat prices"
        
        # RSI should be around 50 (neutral)
        assert 45 < features['RSI'].mean() < 55, "RSI should be neutral for flat prices"
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        # Only 1 row
        df_short = pd.DataFrame({
            'Open': [100], 'High': [101], 'Low': [99], 'Close': [100]
        })
        
        features = compute_features(df_short)
        assert len(features) == 0, "Should return empty DataFrame for insufficient data"
        
        # Only 10 rows (less than minimum required)
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df_short = pd.DataFrame({
            'Open': [100] * 10,
            'High': [101] * 10,
            'Low': [99] * 10,
            'Close': [100] * 10,
        }, index=dates)
        
        features = compute_features(df_short)
        # Should return very few or no rows after dropping NaN
        assert len(features) < 5, "Should have very few rows with insufficient data"
    
    def test_reproducibility(self, sample_ohlc):
        """Test that feature computation is deterministic"""
        features1 = compute_features(sample_ohlc.copy())
        features2 = compute_features(sample_ohlc.copy())
        
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_feature_correlation(self, sample_ohlc):
        """Test that features have reasonable correlation structure"""
        features = compute_features(sample_ohlc)
        
        corr_matrix = features.corr()
        
        # No perfect correlations (except diagonal)
        off_diagonal = corr_matrix.values[~np.eye(5, dtype=bool)]
        assert np.abs(off_diagonal).max() < 0.99, "Features should not be perfectly correlated"
    
    def test_extreme_prices(self):
        """Test handling of extreme price movements"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        # Create data with extreme movements
        prices = [100]
        for _ in range(49):
            # Random walk with occasional large jumps
            change = np.random.randn() * 0.01
            if np.random.random() < 0.1:  # 10% chance of large move
                change *= 10
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
        }, index=dates)
        
        # Should handle without errors
        features = compute_features(df)
        assert len(features) > 0, "Should handle extreme price movements"
        assert features.isnull().sum().sum() == 0, "Should not have NaN values"
    
    def test_missing_columns(self):
        """Test error handling for missing required columns"""
        df = pd.DataFrame({
            'Close': [100, 101, 102]
        })
        
        # compute_features now handles missing columns gracefully
        features = compute_features(df)
        # Should return empty or very limited features due to validation failure
        assert len(features) == 0, "Should return empty features for invalid input"
    
    def test_feature_ranges(self, sample_ohlc):
        """Test that all features are within reasonable ranges"""
        features = compute_features(sample_ohlc)
        
        # Log returns should be small (typically < 10% daily)
        assert np.abs(features['log_return']).max() < 0.5, "Log returns seem unreasonably large"
        
        # Volatility should be positive and reasonable
        assert 0 <= features['volatility'].max() < 1.0, "Volatility should be reasonable"
        
        # RSI between 0 and 100
        assert 0 <= features['RSI'].min() <= features['RSI'].max() <= 100
        
        # MACD should be reasonable (not infinite or extremely large)
        assert np.isfinite(features['MACD']).all(), "MACD should be finite"
        assert np.abs(features['MACD']).max() < 10, "MACD should be reasonable magnitude"
        
        # BBW should be positive and reasonable
        assert 0 <= features['BBW'].min() <= features['BBW'].max() < 1.0


class TestLoadAndComputeFeatures:
    """Test suite for load_and_compute_features function"""
    
    def test_load_csv(self, tmp_path):
        """Test loading and computing features from CSV file"""
        # Create temporary CSV file
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
        })
        
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and compute features
        features, summary = load_and_compute_features(str(csv_path))
        
        assert len(features) > 0, "Should compute features from CSV"
        assert features.shape[1] == 5, "Should have 5 features"
        assert isinstance(summary, dict), "Should return summary dictionary"
        assert summary['validation']['is_valid'], "Features should be valid"
    
    def test_load_parquet(self, tmp_path):
        """Test loading and computing features from Parquet file"""
        # Create temporary Parquet file
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
        }, index=dates)
        
        parquet_path = tmp_path / "test_data.parquet"
        df.to_parquet(parquet_path)
        
        # Load and compute features
        features, raw_df = load_and_compute_features(str(parquet_path))
        
        assert len(features) > 0, "Should compute features from Parquet"
        assert features.shape[1] == 5, "Should have 5 features"
    
    def test_file_not_found(self):
        """Test error handling for non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_and_compute_features("nonexistent_file.csv")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
