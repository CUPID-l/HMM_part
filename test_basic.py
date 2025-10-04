#!/usr/bin/env python3
"""
Simple test to verify the HMM system works.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_basic_functionality():
    """Test basic functionality without complex dependencies."""
    print("Testing HMM Market Regime Detection System...")
    
    try:
        # Test 1: Basic imports
        print("1. Testing imports...")
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        print("   ✓ Core packages imported")
        
        # Test 2: Feature computation
        print("2. Testing feature computation...")
        from features import compute_features
        
        # Generate simple test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, 100))
        
        ohlc_data = pd.DataFrame({
            'Open': prices,
            'High': prices + np.abs(np.random.normal(0, 0.5, 100)),
            'Low': prices - np.abs(np.random.normal(0, 0.5, 100)),
            'Close': prices + np.random.normal(0, 0.1, 100)
        }, index=dates)
        
        features = compute_features(ohlc_data)
        print(f"   ✓ Features computed: {features.shape}")
        print(f"   ✓ Feature columns: {list(features.columns)}")
        
        # Test 3: HMM model (basic test)
        print("3. Testing HMM model...")
        
        try:
            from model_wrapper import HMMRegimeModel
            
            # Use clean data for model
            clean_features = features.dropna()
            if len(clean_features) > 50:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clean_features.values)
                
                model = HMMRegimeModel(n_states=2, random_state=42)
                model.fit(scaled_data)
                
                print(f"   ✓ HMM model trained with {model.n_states} states")
                
                # Test predictions
                states = model.predict(scaled_data)
                probs = model.predict_proba(scaled_data)
                
                print(f"   ✓ Predictions made: {len(states)} states")
                print(f"   ✓ Posterior probabilities: {probs.shape}")
                
                # Test transition matrix
                A = model.get_transition_matrix()
                print(f"   ✓ Transition matrix: {A.shape}")
                
            else:
                print("   ! Insufficient clean data for model training")
                
        except ImportError as e:
            print(f"   ! HMM backend not available: {e}")
            print("   Install hmmlearn: pip install hmmlearn")
            
        # Test 4: Utility functions
        print("4. Testing utility functions...")
        from utils import calculate_performance_metrics, format_currency_pair
        
        # Test performance metrics
        returns = pd.Series(np.random.normal(0.001, 0.02, 50))
        metrics = calculate_performance_metrics(returns)
        print(f"   ✓ Performance metrics: {len(metrics)} calculated")
        
        # Test currency formatting
        formatted = format_currency_pair('EUR/USD')
        assert formatted == 'EUR_USD'
        print("   ✓ Currency pair formatting works")
        
        print("\n" + "="*50)
        print("🎉 BASIC FUNCTIONALITY TEST PASSED!")
        print("="*50)
        print("\nCore system components are working:")
        print("  ✓ Feature engineering")
        print("  ✓ Data processing")  
        print("  ✓ Utility functions")
        
        if 'model' in locals():
            print("  ✓ HMM modeling")
        else:
            print("  ! HMM modeling (needs hmmlearn)")
            
        print("\nSystem is functional for basic operations.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_basic_functionality()
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)
