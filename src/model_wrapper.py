"""
HMM Model Wrapper for Market Regime Detection.

Provides a consistent API over pomegranate or hmmlearn libraries.
Handles model training, prediction, and persistence with proper error handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils import setup_logging, save_model_artifact, set_random_seeds

logger = setup_logging()

try:
    from pomegranate import HiddenMarkovModel, NormalDistribution, State
    from pomegranate import MultivariateGaussianDistribution
    HMM_BACKEND = 'pomegranate'
    logger.info("Using pomegranate backend for HMM")
except ImportError:
    try:
        from hmmlearn.hmm import GaussianHMM
        HMM_BACKEND = 'hmmlearn'
        logger.info("Using hmmlearn backend for HMM")
    except ImportError:
        raise ImportError("Either pomegranate or hmmlearn must be installed")


class HMMRegimeModel:
    """
    Hidden Markov Model wrapper for market regime detection.
    
    Provides consistent API regardless of backend library.
    Supports multivariate Gaussian emissions with full covariance.
    """
    
    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
        init_method: str = "kmeans"
    ):
        """
        Initialize HMM model.
        
        Args:
            n_states: Number of hidden states
            covariance_type: Type of covariance matrix ('full', 'diag', 'tied')
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
            init_method: Initialization method ('kmeans', 'random')
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init_method = init_method
        
        # Model attributes
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.n_features = None
        self.converged = False
        self.n_iter = 0
        self.log_likelihood = None
        
        # Initialize model based on backend
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the underlying HMM model."""
        set_random_seeds(self.random_state)
        
        if HMM_BACKEND == 'hmmlearn':
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
        elif HMM_BACKEND == 'pomegranate':
            # Will be initialized after seeing data dimensions
            self.model = None
        else:
            raise ValueError(f"Unknown HMM backend: {HMM_BACKEND}")
    
    def _initialize_pomegranate_model(self, X: np.ndarray):
        """Initialize pomegranate model after seeing data."""
        n_features = X.shape[1]
        
        # Initialize with k-means clustering
        if self.init_method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Create states with multivariate Gaussian distributions
            states = []
            for i in range(self.n_states):
                mask = labels == i
                if mask.sum() > 1:
                    state_data = X[mask]
                    mean = np.mean(state_data, axis=0)
                    cov = np.cov(state_data.T)
                    
                    # Add regularization to prevent singular covariance
                    cov += np.eye(n_features) * 1e-6
                else:
                    # Fallback for empty clusters
                    mean = np.mean(X, axis=0) + np.random.normal(0, 0.1, n_features)
                    cov = np.eye(n_features) * 0.1
                
                dist = MultivariateGaussianDistribution(mean, cov)
                state = State(dist, name=f"State_{i}")
                states.append(state)
        else:
            # Random initialization
            states = []
            for i in range(self.n_states):
                mean = np.random.normal(0, 1, n_features)
                cov = np.eye(n_features)
                dist = MultivariateGaussianDistribution(mean, cov)
                state = State(dist, name=f"State_{i}")
                states.append(state)
        
        # Create HMM with uniform transition probabilities
        self.model = HiddenMarkovModel.from_states(
            states,
            name="MarketRegimeHMM"
        )
    
    def _prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for training/inference."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check for invalid values
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("Input data contains infinite values")
        
        return X
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'HMMRegimeModel':
        """
        Fit the HMM model to training data.
        
        Args:
            X: Training data (n_samples, n_features)
            feature_names: Optional list of feature names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting HMM model with {self.n_states} states")
        
        X = self._prepare_data(X)
        self.n_features = X.shape[1]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.n_features)]
        
        set_random_seeds(self.random_state)
        
        try:
            if HMM_BACKEND == 'hmmlearn':
                self.model.fit(X)
                self.converged = self.model.monitor_.converged
                self.n_iter = self.model.monitor_.iter
                self.log_likelihood = self.model.score(X)
                
            elif HMM_BACKEND == 'pomegranate':
                # Initialize model if not done yet
                if self.model is None:
                    self._initialize_pomegranate_model(X)
                
                # Fit model
                self.model.fit([X], algorithm='baum-welch', max_iterations=self.max_iter)
                self.converged = True  # Pomegranate doesn't expose convergence info easily
                self.log_likelihood = self.model.log_probability(X)
            
            self.is_fitted = True
            logger.info(f"Model fitted. Log-likelihood: {self.log_likelihood:.4f}")
            
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            self.is_fitted = False
            raise
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence (Viterbi decoding).
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Array of predicted states
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._prepare_data(X)
        
        if HMM_BACKEND == 'hmmlearn':
            states = self.model.predict(X)
        elif HMM_BACKEND == 'pomegranate':
            # Pomegranate returns (log_prob, path)
            _, states = self.model.viterbi(X)
            states = np.array([s[1].name.split('_')[1] for s in states[1:-1]], dtype=int)
        
        return states
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities for each state (forward-backward).
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Array of shape (n_samples, n_states) with posterior probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._prepare_data(X)
        
        if HMM_BACKEND == 'hmmlearn':
            log_proba = self.model.predict_proba(X)
            return log_proba
        elif HMM_BACKEND == 'pomegranate':
            # Use forward-backward algorithm
            log_proba = self.model.forward_backward(X)
            # Convert to probabilities and extract state probabilities
            proba = np.exp(log_proba[1])  # Forward probabilities
            return proba
    
    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of data under the model.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Log-likelihood score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X = self._prepare_data(X)
        
        if HMM_BACKEND == 'hmmlearn':
            return self.model.score(X)
        elif HMM_BACKEND == 'pomegranate':
            return self.model.log_probability(X)
    
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get transition probability matrix A where A[i,j] = P(S_{t+1}=j | S_t=i).
        
        Returns:
            Transition matrix of shape (n_states, n_states)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        if HMM_BACKEND == 'hmmlearn':
            return self.model.transmat_
        elif HMM_BACKEND == 'pomegranate':
            # Extract transition matrix from pomegranate model
            n_states = len(self.model.states) - 2  # Exclude start/end states
            trans_matrix = np.zeros((n_states, n_states))
            
            for i, state_i in enumerate(self.model.states[1:-1]):  # Skip start/end
                for j, state_j in enumerate(self.model.states[1:-1]):
                    # Get transition probability
                    edge_prob = 0.0
                    for edge in self.model.edges:
                        if edge[0] == state_i and edge[1] == state_j:
                            edge_prob = np.exp(edge[2])  # Convert from log prob
                            break
                    trans_matrix[i, j] = edge_prob
            
            return trans_matrix
    
    def get_start_probabilities(self) -> np.ndarray:
        """Get initial state probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        if HMM_BACKEND == 'hmmlearn':
            return self.model.startprob_
        elif HMM_BACKEND == 'pomegranate':
            # Extract start probabilities
            start_probs = np.zeros(self.n_states)
            for i, state in enumerate(self.model.states[1:-1]):
                # Get probability from start state
                for edge in self.model.edges:
                    if edge[0] == self.model.start and edge[1] == state:
                        start_probs[i] = np.exp(edge[2])
                        break
            return start_probs
    
    def get_emission_params(self) -> Dict[str, np.ndarray]:
        """
        Get emission distribution parameters.
        
        Returns:
            Dictionary with 'means' and 'covars' arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting parameters")
        
        if HMM_BACKEND == 'hmmlearn':
            return {
                'means': self.model.means_,
                'covars': self.model.covars_
            }
        elif HMM_BACKEND == 'pomegranate':
            means = []
            covars = []
            for state in self.model.states[1:-1]:  # Skip start/end states
                dist = state.distribution
                means.append(dist.parameters[0])  # Mean
                covars.append(dist.parameters[1])  # Covariance
            
            return {
                'means': np.array(means),
                'covars': np.array(covars)
            }
    
    def compute_next_step_probabilities(
        self, 
        current_posterior: np.ndarray = None,
        current_state: int = None
    ) -> np.ndarray:
        """
        Compute next-step state probabilities.
        
        Args:
            current_posterior: Current posterior probabilities (optional)
            current_state: Current most likely state (optional)
            
        Returns:
            Next-step state probabilities
        """
        transition_matrix = self.get_transition_matrix()
        
        if current_posterior is not None:
            # P(S_{t+1}=j) = sum_i P(S_t=i | obs) * A[i,j]
            return current_posterior @ transition_matrix
        elif current_state is not None:
            # P(S_{t+1}=j) = A[current_state, j]
            return transition_matrix[current_state]
        else:
            raise ValueError("Either current_posterior or current_state must be provided")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        if not self.is_fitted:
            return {'fitted': False}
        
        info = {
            'fitted': True,
            'n_states': self.n_states,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'covariance_type': self.covariance_type,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'converged': self.converged,
            'n_iter': getattr(self, 'n_iter', None),
            'log_likelihood': self.log_likelihood,
            'backend': HMM_BACKEND,
            'transition_matrix': self.get_transition_matrix(),
            'start_probabilities': self.get_start_probabilities(),
            'emission_params': self.get_emission_params()
        }
        
        return info
    
    def save(self, filepath: str, metadata: Dict[str, Any] = None):
        """
        Save model with metadata.
        
        Args:
            filepath: Path to save model
            metadata: Additional metadata to save with model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_info = self.get_model_info()
        if metadata:
            model_info.update(metadata)
        
        save_model_artifact(
            model=self,
            scaler=None,  # Scaler is handled separately
            metadata=model_info,
            filepath=filepath
        )
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'HMMRegimeModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded HMMRegimeModel instance
        """
        artifact = joblib.load(filepath)
        model_instance = artifact['model']
        
        logger.info(f"Model loaded from {filepath}")
        return model_instance


def select_best_model(
    X: np.ndarray,
    n_states_range: List[int] = [2, 3, 4, 5],
    n_runs: int = 5,
    **model_params
) -> Tuple[HMMRegimeModel, Dict]:
    """
    Model selection using cross-validation and information criteria.
    
    Args:
        X: Training data
        n_states_range: Range of states to test
        n_runs: Number of random initializations per configuration
        **model_params: Additional model parameters
        
    Returns:
        Tuple of (best_model, selection_results)
    """
    logger.info(f"Running model selection over {n_states_range} states with {n_runs} runs each")
    
    results = []
    best_model = None
    best_bic = np.inf
    
    for n_states in n_states_range:
        state_results = []
        
        for run in range(n_runs):
            try:
                # Different random seed for each run
                model = HMMRegimeModel(
                    n_states=n_states,
                    random_state=42 + run,
                    **model_params
                )
                
                model.fit(X)
                
                # Calculate information criteria
                log_likelihood = model.score(X)
                n_params = n_states * (n_states - 1) + n_states * X.shape[1] * (X.shape[1] + 1) / 2
                aic = -2 * log_likelihood + 2 * n_params
                bic = -2 * log_likelihood + np.log(len(X)) * n_params
                
                run_result = {
                    'n_states': n_states,
                    'run': run,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'bic': bic,
                    'converged': model.converged,
                    'model': model
                }
                
                state_results.append(run_result)
                
                # Track best model by BIC
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                
            except Exception as e:
                logger.warning(f"Failed to fit model with {n_states} states, run {run}: {e}")
                continue
        
        results.extend(state_results)
        
        if state_results:
            best_run = min(state_results, key=lambda x: x['bic'])
            logger.info(f"n_states={n_states}: Best BIC={best_run['bic']:.2f}, "
                       f"Best LL={best_run['log_likelihood']:.2f}")
    
    selection_summary = {
        'all_results': results,
        'best_n_states': best_model.n_states if best_model else None,
        'best_bic': best_bic if best_model else None,
        'selection_criteria': 'BIC'
    }
    
    logger.info(f"Model selection complete. Best model: {best_model.n_states} states, "
               f"BIC: {best_bic:.2f}")
    
    return best_model, selection_summary


if __name__ == "__main__":
    # Example usage and testing
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    print(f"Testing HMM implementation with {HMM_BACKEND} backend")
    print(f"Data shape: {X.shape}")
    
    # Test basic model
    model = HMMRegimeModel(n_states=3, random_state=42)
    model.fit(X)
    
    # Test predictions
    states = model.predict(X)
    posteriors = model.predict_proba(X)
    score = model.score(X)
    
    print(f"Fitted model with {model.n_states} states")
    print(f"Log-likelihood: {score:.4f}")
    print(f"State distribution: {np.bincount(states)}")
    
    # Test model selection
    best_model, selection_results = select_best_model(
        X, 
        n_states_range=[2, 3, 4], 
        n_runs=3
    )
    
    print(f"Best model: {best_model.n_states} states")
    print("Model selection complete")
