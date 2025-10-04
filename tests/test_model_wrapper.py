"""
Unit tests for HMM model wrapper.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_wrapper import HMMRegimeModel


class TestHMMRegimeModel:
    """Test suite for HMMRegimeModel class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 200
        n_features = 5
        
        # Generate data from 3 distinct regimes
        regime1 = np.random.randn(70, n_features) * 0.5 + np.array([0.02, 0.01, 60, 0.01, 0.02])
        regime2 = np.random.randn(60, n_features) * 0.3 + np.array([0.00, 0.005, 50, 0.00, 0.015])
        regime3 = np.random.randn(70, n_features) * 0.7 + np.array([-0.02, 0.015, 40, -0.01, 0.025])
        
        X = np.vstack([regime1, regime2, regime3])
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        
        return X
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = HMMRegimeModel(n_states=3, random_state=42)
        
        assert model.n_states == 3
        assert model.random_state == 42
        assert not model.is_fitted
    
    def test_model_fit(self, sample_data):
        """Test model fitting"""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(sample_data)
        
        assert model.is_fitted, "Model should be fitted after fit()"
        assert model.converged, "Model should converge on well-behaved data"
        assert hasattr(model, 'n_iter'), "Should track number of iterations"
    
    def test_predict(self, sample_data):
        """Test prediction (Viterbi decoding)"""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(sample_data)
        
        states = model.predict(sample_data)
        
        assert len(states) == len(sample_data), "Should predict for all samples"
        assert states.min() >= 0, "States should be non-negative"
        assert states.max() < 3, "States should be less than n_states"
        assert len(np.unique(states)) > 1, "Should use multiple states"
    
    def test_predict_proba(self, sample_data):
        """Test posterior probability computation"""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(sample_data)
        
        posteriors = model.predict_proba(sample_data)
        
        assert posteriors.shape == (len(sample_data), 3), "Should have shape (n_samples, n_states)"
        assert np.allclose(posteriors.sum(axis=1), 1.0), "Posteriors should sum to 1"
        assert (posteriors >= 0).all(), "Posteriors should be non-negative"
        assert (posteriors <= 1).all(), "Posteriors should be <= 1"
    
    def test_score(self, sample_data):
        """Test log-likelihood computation"""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(sample_data)
        
        log_likelihood = model.score(sample_data)
        
        assert isinstance(log_likelihood, float), "Score should return float"
        assert log_likelihood < 0, "Log-likelihood should be negative"
        assert np.isfinite(log_likelihood), "Log-likelihood should be finite"
    
    def test_get_transition_matrix(self, sample_data):
        """Test transition matrix extraction"""
        model = HMMRegimeModel(n_states=3, random_state=42)
        model.fit(sample_data)
        
        A = model.get_transition_matrix()
        
        assert A.shape == (3, 3), "Transition matrix should be n_states x n_states"
        assert np.allclose(A.sum(axis=1), 1.0), "Rows should sum to 1"
        assert (A >= 0).all(), "Probabilities should be non-negative"
        assert (A <= 1).all(), "Probabilities should be <= 1"
    
    def test_reproducibility(self, sample_data):
        """Test that same seed gives same results"""
        model1 = HMMRegimeModel(n_states=3, random_state=42)
        model1.fit(sample_data)
        states1 = model1.predict(sample_data)
        
        model2 = HMMRegimeModel(n_states=3, random_state=42)
        model2.fit(sample_data)
        states2 = model2.predict(sample_data)
        
        np.testing.assert_array_equal(states1, states2)
    
    def test_different_n_states(self, sample_data):
        """Test models with different numbers of states"""
        for n_states in [2, 3, 4]:
            model = HMMRegimeModel(n_states=n_states, random_state=42)
            model.fit(sample_data)
            
            states = model.predict(sample_data)
            assert states.max() < n_states, f"States should be < {n_states}"
            
            A = model.get_transition_matrix()
            assert A.shape == (n_states, n_states)
    
    def test_predict_before_fit(self, sample_data):
        """Test error handling when predicting before fitting"""
        model = HMMRegimeModel(n_states=3)
        
        with pytest.raises(ValueError):
            model.predict(sample_data)
    
    def test_small_dataset(self):
        """Test behavior with very small dataset"""
        X = np.random.randn(10, 5)
        model = HMMRegimeModel(n_states=3, random_state=42)
        
        # Should either fit or raise appropriate error
        try:
            model.fit(X)
            # If it fits, predictions should work
            states = model.predict(X)
            assert len(states) == 10
        except (ValueError, RuntimeError):
            # Acceptable to fail with insufficient data
            pass
    
    def test_single_feature(self):
        """Test with single feature"""
        X = np.random.randn(100, 1)
        model = HMMRegimeModel(n_states=2, random_state=42)
        model.fit(X)
        
        states = model.predict(X)
        assert len(states) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
