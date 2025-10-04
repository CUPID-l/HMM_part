"""
Model selection utilities for HMM.

Provides AIC/BIC computation and automatic model selection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from model_wrapper import HMMRegimeModel

logger = logging.getLogger(__name__)


def compute_aic_bic(
    model: HMMRegimeModel, 
    X: np.ndarray, 
    n_states: int, 
    n_features: int
) -> Dict[str, float]:
    """
    Compute AIC and BIC for HMM model selection.
    
    AIC = -2*log(L) + 2*k
    BIC = -2*log(L) + k*log(n)
    
    where:
        L = likelihood
        k = number of parameters
        n = number of observations
    
    Parameters for Gaussian HMM:
        - Transition matrix: n_states * (n_states - 1) [rows sum to 1]
        - Start probabilities: n_states - 1 [sum to 1]
        - Emission means: n_states * n_features
        - Emission covariances: n_states * n_features * (n_features + 1) / 2 [full covariance]
    
    Args:
        model: Fitted HMMRegimeModel
        X: Training data
        n_states: Number of states
        n_features: Number of features
        
    Returns:
        Dict with log_likelihood, aic, bic, n_params
    """
    # Compute log-likelihood
    log_likelihood = model.score(X)
    n_samples = len(X)
    
    # Count parameters
    # Transition matrix: each row has n_states-1 free parameters (last is 1-sum)
    n_transition_params = n_states * (n_states - 1)
    
    # Start probabilities: n_states-1 free parameters
    n_start_params = n_states - 1
    
    # Emission means: n_states * n_features
    n_emission_means = n_states * n_features
    
    # Emission covariances (full covariance matrix):
    # For each state: n_features*(n_features+1)/2 parameters (symmetric matrix)
    n_emission_covs = n_states * n_features * (n_features + 1) // 2
    
    n_params = n_transition_params + n_start_params + n_emission_means + n_emission_covs
    
    # Compute information criteria
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    
    return {
        'log_likelihood': log_likelihood,
        'aic': aic,
        'bic': bic,
        'n_params': n_params,
        'n_samples': n_samples
    }


def select_best_n_states(
    X: np.ndarray,
    n_states_range: List[int] = None,
    n_trials: int = 5,
    criterion: str = 'bic',
    random_state: int = 42,
    **model_kwargs
) -> Dict:
    """
    Train HMMs with different numbers of states and select best.
    
    Runs multiple trials per K to handle random initialization,
    then selects the best model based on AIC or BIC.
    
    Args:
        X: Training data (n_samples, n_features)
        n_states_range: List of n_states values to try (default: [2, 3, 4, 5])
        n_trials: Number of random initializations per n_states
        criterion: 'aic' or 'bic' for model selection
        random_state: Base random seed
        **model_kwargs: Additional arguments for HMMRegimeModel
        
    Returns:
        Dict with:
            - best_model: Best HMMRegimeModel
            - best_n_states: Optimal number of states
            - best_score: Best AIC/BIC score
            - all_results: List of all trials with metrics
    """
    if n_states_range is None:
        n_states_range = [2, 3, 4, 5]
    
    if criterion not in ['aic', 'bic']:
        raise ValueError("criterion must be 'aic' or 'bic'")
    
    logger.info(f"Model selection: testing n_states in {n_states_range} with {n_trials} trials each")
    logger.info(f"Selection criterion: {criterion.upper()}")
    
    all_results = []
    n_features = X.shape[1]
    
    for n_states in n_states_range:
        logger.info(f"Testing n_states={n_states}...")
        
        best_score = np.inf
        best_model = None
        best_metrics = None
        
        for trial in range(n_trials):
            try:
                # Train model with different random seed
                seed = random_state + trial + n_states * 1000
                model = HMMRegimeModel(
                    n_states=n_states,
                    random_state=seed,
                    **model_kwargs
                )
                model.fit(X)
                
                # Compute metrics
                metrics = compute_aic_bic(model, X, n_states, n_features)
                score = metrics[criterion]
                
                # Keep best model for this n_states
                if score < best_score:
                    best_score = score
                    best_model = model
                    best_metrics = metrics
                
                logger.debug(
                    f"  Trial {trial+1}/{n_trials}: "
                    f"LL={metrics['log_likelihood']:.2f}, "
                    f"AIC={metrics['aic']:.2f}, "
                    f"BIC={metrics['bic']:.2f}"
                )
                
            except Exception as e:
                logger.warning(f"  Trial {trial+1}/{n_trials} failed: {e}")
                continue
        
        if best_model is not None:
            all_results.append({
                'n_states': n_states,
                'model': best_model,
                **best_metrics,
                'criterion_score': best_score
            })
            
            logger.info(
                f"n_states={n_states}: "
                f"Best {criterion.upper()}={best_score:.2f}, "
                f"LL={best_metrics['log_likelihood']:.2f}, "
                f"Converged={best_model.converged}"
            )
        else:
            logger.warning(f"All trials failed for n_states={n_states}")
    
    if not all_results:
        raise RuntimeError("All model training attempts failed")
    
    # Select model with lowest criterion score
    best_result = min(all_results, key=lambda x: x['criterion_score'])
    
    logger.info(
        f"\n{'='*60}\n"
        f"SELECTED: n_states={best_result['n_states']} "
        f"with {criterion.upper()}={best_result['criterion_score']:.2f}\n"
        f"{'='*60}"
    )
    
    return {
        'best_model': best_result['model'],
        'best_n_states': best_result['n_states'],
        'best_score': best_result['criterion_score'],
        'best_metrics': {k: v for k, v in best_result.items() 
                        if k not in ['model', 'n_states', 'criterion_score']},
        'all_results': all_results,
        'criterion': criterion
    }


def compare_models(
    X: np.ndarray,
    models: List[HMMRegimeModel],
    n_states_list: List[int],
    labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple fitted HMM models using information criteria.
    
    Args:
        X: Training data
        models: List of fitted HMMRegimeModel objects
        n_states_list: List of n_states for each model
        labels: Optional labels for each model
        
    Returns:
        DataFrame with comparison metrics
    """
    import pandas as pd
    
    if labels is None:
        labels = [f"Model_{i+1}" for i in range(len(models))]
    
    n_features = X.shape[1]
    
    results = []
    for i, (model, n_states, label) in enumerate(zip(models, n_states_list, labels)):
        metrics = compute_aic_bic(model, X, n_states, n_features)
        
        results.append({
            'Model': label,
            'n_states': n_states,
            'Log-Likelihood': metrics['log_likelihood'],
            'AIC': metrics['aic'],
            'BIC': metrics['bic'],
            'n_params': metrics['n_params'],
            'Converged': model.converged
        })
    
    df = pd.DataFrame(results)
    
    # Add delta columns
    df['Δ_AIC'] = df['AIC'] - df['AIC'].min()
    df['Δ_BIC'] = df['BIC'] - df['BIC'].min()
    
    # Sort by BIC
    df = df.sort_values('BIC').reset_index(drop=True)
    
    return df


def cross_validate_hmm(
    X: np.ndarray,
    n_states: int,
    n_folds: int = 5,
    random_state: int = 42,
    **model_kwargs
) -> Dict:
    """
    Perform time-series cross-validation for HMM.
    
    Uses expanding window approach suitable for time series:
    - Fold 1: Train on [0:T/5], test on [T/5:2T/5]
    - Fold 2: Train on [0:2T/5], test on [2T/5:3T/5]
    - etc.
    
    Args:
        X: Training data
        n_states: Number of HMM states
        n_folds: Number of folds
        random_state: Random seed
        **model_kwargs: Additional arguments for HMMRegimeModel
        
    Returns:
        Dict with cross-validation results
    """
    n_samples = len(X)
    fold_size = n_samples // (n_folds + 1)
    
    logger.info(f"Cross-validating HMM with {n_states} states using {n_folds} folds")
    
    cv_scores = []
    
    for fold in range(n_folds):
        # Expanding window
        train_end = (fold + 2) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)
        
        X_train = X[:train_end]
        X_test = X[test_start:test_end]
        
        logger.info(f"Fold {fold+1}/{n_folds}: Train [0:{train_end}], Test [{test_start}:{test_end}]")
        
        # Train model
        model = HMMRegimeModel(
            n_states=n_states,
            random_state=random_state + fold,
            **model_kwargs
        )
        model.fit(X_train)
        
        # Evaluate on test set
        test_ll = model.score(X_test)
        train_ll = model.score(X_train)
        
        cv_scores.append({
            'fold': fold + 1,
            'train_ll': train_ll,
            'test_ll': test_ll,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })
        
        logger.info(f"  Train LL: {train_ll:.2f}, Test LL: {test_ll:.2f}")
    
    # Aggregate results
    mean_train_ll = np.mean([s['train_ll'] for s in cv_scores])
    mean_test_ll = np.mean([s['test_ll'] for s in cv_scores])
    std_test_ll = np.std([s['test_ll'] for s in cv_scores])
    
    logger.info(
        f"CV Results: Mean Test LL = {mean_test_ll:.2f} ± {std_test_ll:.2f}, "
        f"Mean Train LL = {mean_train_ll:.2f}"
    )
    
    return {
        'mean_train_ll': mean_train_ll,
        'mean_test_ll': mean_test_ll,
        'std_test_ll': std_test_ll,
        'cv_scores': cv_scores,
        'n_folds': n_folds
    }
