"""
Unit tests for the HMM market regime detection system.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from features import compute_features, validate_features
from model_wrapper import HMMRegimeModel
from utils import (
    validate_ohlc_data, clean_ohlc_data, label_states_by_return_volatility,
    calculate_performance_metrics, format_currency_pair
)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions."""
    
    def setUp(self):
        """Create sample OHLC data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        
        # Generate realistic OHLC data
        returns = np.random.normal(0, 0.01, 300)
        prices = 100 * (1 + returns).cumprod()
        
        noise = np.random.normal(0, 0.002, (300, 3))  # For high, low, open
        
        self.ohlc_data = pd.DataFrame({
            'Open': prices + noise[:, 0],
            'High': prices + abs(noise[:, 1]) + 0.001,
            'Low': prices - abs(noise[:, 2]) - 0.001,
            'Close': prices
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        self.ohlc_data['High'] = np.maximum(
            self.ohlc_data['High'],
            self.ohlc_data[['Open', 'Close']].max(axis=1)
        )
        self.ohlc_data['Low'] = np.minimum(
            self.ohlc_data['Low'],
            self.ohlc_data[['Open', 'Close']].min(axis=1)
        )
    
    def test_validate_ohlc_data(self):
        """Test OHLC data validation."""
        self.assertTrue(validate_ohlc_data(self.ohlc_data))
        
        # Test invalid data
        invalid_data = self.ohlc_data.copy()
        invalid_data.loc[invalid_data.index[0], 'High'] = invalid_data.loc[invalid_data.index[0], 'Low'] - 1
        self.assertFalse(validate_ohlc_data(invalid_data))
    
    def test_compute_features(self):
        """Test feature computation."""
        features = compute_features(self.ohlc_data)
        
        # Check that all required features are present
        expected_features = ['log_return', 'volatility', 'RSI', 'MACD', 'BBW']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Check data types
        self.assertTrue(features.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all())
        
        # Check for reasonable ranges
        self.assertTrue((features['RSI'] >= 0).all() and (features['RSI'] <= 100).all())
        self.assertTrue((features['volatility'] >= 0).all())
        self.assertTrue((features['BBW'] > 0).all())
    
    def test_validate_features(self):
        """Test feature validation."""
        features = compute_features(self.ohlc_data)
        is_valid, message = validate_features(features)
        self.assertTrue(is_valid, f"Feature validation failed: {message}")


class TestHMMModel(unittest.TestCase):
    """Test HMM model wrapper."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 5
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = HMMRegimeModel(n_states=3, random_state=42)
        self.assertEqual(model.n_states, 3)
        self.assertFalse(model.is_fitted)
    
    def test_model_fitting(self):
        """Test model fitting."""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(self.X, feature_names=self.feature_names)
        
        self.assertTrue(model.is_fitted)
        self.assertEqual(model.n_features, self.n_features)
        self.assertEqual(model.feature_names, self.feature_names)
    
    def test_model_prediction(self):
        """Test model prediction methods."""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(self.X)
        
        # Test Viterbi prediction
        states = model.predict(self.X)
        self.assertEqual(len(states), self.n_samples)
        self.assertTrue(np.all(states >= 0) and np.all(states < 3))
        
        # Test posterior probabilities
        posteriors = model.predict_proba(self.X)
        self.assertEqual(posteriors.shape, (self.n_samples, 3))
        self.assertTrue(np.allclose(posteriors.sum(axis=1), 1.0))
    
    def test_transition_matrix(self):
        """Test transition matrix extraction."""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(self.X)
        
        A = model.get_transition_matrix()
        self.assertEqual(A.shape, (3, 3))
        self.assertTrue(np.allclose(A.sum(axis=1), 1.0))  # Rows sum to 1
        self.assertTrue(np.all(A >= 0))  # All probabilities non-negative
    
    def test_next_step_probabilities(self):
        """Test next-step probability computation."""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(self.X)
        
        # Test with current state
        next_probs_state = model.compute_next_step_probabilities(current_state=0)
        self.assertEqual(len(next_probs_state), 3)
        self.assertAlmostEqual(sum(next_probs_state), 1.0)
        
        # Test with posterior distribution
        current_posterior = np.array([0.5, 0.3, 0.2])
        next_probs_post = model.compute_next_step_probabilities(current_posterior=current_posterior)
        self.assertEqual(len(next_probs_post), 3)
        self.assertAlmostEqual(sum(next_probs_post), 1.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_format_currency_pair(self):
        """Test currency pair formatting."""
        self.assertEqual(format_currency_pair('EUR/USD'), 'EUR_USD')
        self.assertEqual(format_currency_pair('eur-usd'), 'EUR_USD')
        self.assertEqual(format_currency_pair('XAUUSD'), 'XAUUSD')
    
    def test_label_states_by_return_volatility(self):
        """Test state labeling function."""
        # Create mock data
        states = np.array([0, 0, 1, 1, 2, 2])
        returns = pd.Series([0.01, 0.02, -0.01, -0.02, 0.001, 0.002])
        
        labels = label_states_by_return_volatility(states, returns)
        
        self.assertIsInstance(labels, dict)
        self.assertEqual(len(labels), 3)
        self.assertIn('Bull', labels.values())
        self.assertIn('Bear', labels.values())
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create mock return series
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.002])
        
        metrics = calculate_performance_metrics(returns)
        
        required_metrics = ['total_return', 'cagr', 'annual_volatility', 
                           'sharpe_ratio', 'max_drawdown', 'hit_rate']
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_clean_ohlc_data(self):
        """Test OHLC data cleaning."""
        # Create data with issues
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        dirty_data = pd.DataFrame({
            'Open': [100, 101, np.nan, 103, 104, 0, 106, 107, 108, 109],
            'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Low': [99, 100, 101, 102, 103, -1, 105, 106, 107, 108],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5]
        }, index=dates)
        
        cleaned_data = clean_ohlc_data(dirty_data)
        
        # Should remove rows with NaN, zero, or negative prices
        self.assertTrue(len(cleaned_data) < len(dirty_data))
        self.assertFalse(cleaned_data.isna().any().any())
        self.assertTrue((cleaned_data > 0).all().all())


class TestModelPersistence(unittest.TestCase):
    """Test model saving and loading."""
    
    def setUp(self):
        """Set up test model."""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.model = HMMRegimeModel(n_states=2, random_state=42)
        self.model.fit(self.X)
    
    def test_model_save_load(self):
        """Test model save and load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, 'test_model.joblib')
            self.model.save(model_path, metadata={'test': 'data'})
            
            # Load model
            loaded_model = HMMRegimeModel.load(model_path)
            
            # Verify loaded model
            self.assertTrue(loaded_model.is_fitted)
            self.assertEqual(loaded_model.n_states, 2)
            
            # Test predictions are the same
            original_pred = self.model.predict(self.X)
            loaded_pred = loaded_model.predict(self.X)
            np.testing.assert_array_equal(original_pred, loaded_pred)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from data to prediction."""
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        
        returns = np.random.normal(0, 0.01, 500)
        prices = 100 * (1 + returns).cumprod()
        noise = np.random.normal(0, 0.002, (500, 3))
        
        ohlc_data = pd.DataFrame({
            'Open': prices + noise[:, 0],
            'High': prices + abs(noise[:, 1]) + 0.001,
            'Low': prices - abs(noise[:, 2]) - 0.001,
            'Close': prices
        }, index=dates)
        
        # Ensure valid OHLC
        ohlc_data['High'] = np.maximum(
            ohlc_data['High'],
            ohlc_data[['Open', 'Close']].max(axis=1)
        )
        ohlc_data['Low'] = np.minimum(
            ohlc_data['Low'],
            ohlc_data[['Open', 'Close']].min(axis=1)
        )
        
        # Compute features
        features = compute_features(ohlc_data)
        self.assertGreater(len(features), 0)
        
        # Fit model
        model = HMMRegimeModel(n_states=3, random_state=42)
        
        # Use subset for training
        train_data = features.iloc[:400].dropna()
        model.fit(train_data.values, feature_names=list(train_data.columns))
        
        # Make predictions
        test_data = features.iloc[400:].dropna()
        if len(test_data) > 0:
            states = model.predict(test_data.values)
            posteriors = model.predict_proba(test_data.values)
            
            self.assertEqual(len(states), len(test_data))
            self.assertEqual(posteriors.shape[0], len(test_data))
            self.assertEqual(posteriors.shape[1], 3)


if __name__ == '__main__':
    # Set up test environment
    import warnings
    warnings.filterwarnings('ignore')
    
    # Run tests
    unittest.main(verbosity=2)
