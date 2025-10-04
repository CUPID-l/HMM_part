"""
Unit tests for data quality checks.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_quality import DataQualityChecker


class TestOHLCConsistency:
    """Test OHLC relationship validation"""
    
    @pytest.fixture
    def valid_ohlc(self):
        """Create valid OHLC data"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        }, index=dates)
        return df
    
    def test_valid_data(self, valid_ohlc):
        """Test that valid OHLC data passes"""
        checker = DataQualityChecker()
        result = checker.check_ohlc_consistency(valid_ohlc)
        
        assert result['valid'], "Valid OHLC data should pass"
        assert len(result['issues']) == 0, "Should have no issues"
    
    def test_high_less_than_low(self):
        """Test detection of High < Low"""
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [99, 102],  # First row: High < Low
            'Low': [100, 101],
            'Close': [100, 101],
        })
        
        checker = DataQualityChecker()
        result = checker.check_ohlc_consistency(df)
        
        assert not result['valid'], "Should detect High < Low"
        assert any('High < Low' in issue for issue in result['issues'])
    
    def test_high_less_than_open(self):
        """Test detection of High < Open"""
        df = pd.DataFrame({
            'Open': [105, 101],  # First row: Open > High
            'High': [104, 102],
            'Low': [99, 100],
            'Close': [100, 101],
        })
        
        checker = DataQualityChecker()
        result = checker.check_ohlc_consistency(df)
        
        assert not result['valid'], "Should detect High < Open"
        assert any('High < Open' in issue for issue in result['issues'])
    
    def test_high_less_than_close(self):
        """Test detection of High < Close"""
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [104, 102],
            'Low': [99, 100],
            'Close': [105, 101],  # First row: Close > High
        })
        
        checker = DataQualityChecker()
        result = checker.check_ohlc_consistency(df)
        
        assert not result['valid'], "Should detect High < Close"
        assert any('High < Close' in issue for issue in result['issues'])
    
    def test_low_greater_than_open(self):
        """Test detection of Low > Open"""
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [105, 106],
            'Low': [101, 100],  # First row: Low > Open
            'Close': [102, 103],
        })
        
        checker = DataQualityChecker()
        result = checker.check_ohlc_consistency(df)
        
        assert not result['valid'], "Should detect Low > Open"
        assert any('Low > Open' in issue for issue in result['issues'])


class TestMissingData:
    """Test missing data detection"""
    
    def test_no_missing_data(self):
        """Test that complete data passes"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': range(10),
            'High': range(1, 11),
            'Low': range(10),
            'Close': range(10),
        }, index=dates)
        
        checker = DataQualityChecker()
        result = checker.check_missing_data(df)
        
        assert result['valid'], "Complete data should pass"
    
    def test_nan_values(self):
        """Test detection of NaN values"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': range(1, 11),
            'Low': range(10),
            'Close': range(10),
        }, index=dates)
        
        checker = DataQualityChecker()
        result = checker.check_missing_data(df)
        
        assert not result['valid'], "Should detect NaN values"
        assert any('NaN' in issue for issue in result['issues'])
    
    def test_date_gaps(self):
        """Test detection of large date gaps"""
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-10'])  # 8-day gap
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100, 101, 102],
        }, index=dates)
        
        checker = DataQualityChecker()
        result = checker.check_missing_data(df, max_gap_days=5)
        
        assert not result['valid'], "Should detect large date gap"
        assert any('gaps' in issue.lower() for issue in result['issues'])
    
    def test_non_datetime_index(self):
        """Test handling of non-datetime index"""
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100, 101, 102],
        })
        
        checker = DataQualityChecker()
        result = checker.check_missing_data(df)
        
        assert not result['valid'], "Should detect non-datetime index"


class TestPriceSanity:
    """Test price sanity checks"""
    
    def test_normal_prices(self):
        """Test that normal prices pass"""
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103],
            'High': [102, 103, 104, 105],
            'Low': [99, 100, 101, 102],
            'Close': [101, 102, 103, 104],
        })
        
        checker = DataQualityChecker()
        result = checker.check_price_sanity(df)
        
        assert result['valid'], "Normal prices should pass"
    
    def test_extreme_returns(self):
        """Test detection of extreme price movements"""
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103],
            'High': [102, 103, 104, 105],
            'Low': [99, 100, 101, 102],
            'Close': [100, 101, 150, 104],  # 48% jump on third day
        })
        
        checker = DataQualityChecker()
        result = checker.check_price_sanity(df, max_daily_change=0.20)
        
        assert not result['valid'], "Should detect extreme price change"
        assert any('price change' in issue.lower() for issue in result['issues'])
    
    def test_zero_prices(self):
        """Test detection of zero prices"""
        df = pd.DataFrame({
            'Open': [100, 0, 102, 103],  # Zero price
            'High': [102, 103, 104, 105],
            'Low': [99, 100, 101, 102],
            'Close': [100, 101, 102, 103],
        })
        
        checker = DataQualityChecker()
        result = checker.check_price_sanity(df)
        
        assert not result['valid'], "Should detect zero prices"
        assert any('zero' in issue.lower() or 'negative' in issue.lower() 
                  for issue in result['issues'])
    
    def test_negative_prices(self):
        """Test detection of negative prices"""
        df = pd.DataFrame({
            'Open': [100, 101, -5, 103],  # Negative price
            'High': [102, 103, 104, 105],
            'Low': [99, 100, 101, 102],
            'Close': [100, 101, 102, 103],
        })
        
        checker = DataQualityChecker()
        result = checker.check_price_sanity(df)
        
        assert not result['valid'], "Should detect negative prices"


class TestFullValidation:
    """Test complete validation pipeline"""
    
    def test_all_valid(self):
        """Test that fully valid data passes all checks"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        close_prices = 100 * np.exp((np.random.randn(50) * 0.01).cumsum())
        
        df = pd.DataFrame({
            'Open': close_prices * (1 + np.random.randn(50) * 0.002),
            'High': close_prices * (1 + np.abs(np.random.randn(50)) * 0.003),
            'Low': close_prices * (1 - np.abs(np.random.randn(50)) * 0.003),
            'Close': close_prices,
        }, index=dates)
        
        # Ensure OHLC consistency
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        checker = DataQualityChecker()
        result = checker.validate_data(df)
        
        assert result['all_valid'], "Valid data should pass all checks"
        assert all(check['valid'] for check in result['checks'].values())
    
    def test_multiple_issues(self):
        """Test detection of multiple issues"""
        dates = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-20'])  # Gap
        df = pd.DataFrame({
            'Open': [100, np.nan, 102],  # NaN
            'High': [99, 103, 104],  # First row: High < Low
            'Low': [100, 100, 101],
            'Close': [100, 101, 102],
        }, index=dates)
        
        checker = DataQualityChecker()
        result = checker.validate_data(df)
        
        assert not result['all_valid'], "Should detect multiple issues"
        
        # Should have at least 2 failed checks
        failed_checks = [name for name, check in result['checks'].items() 
                        if not check['valid']]
        assert len(failed_checks) >= 2, "Should detect multiple types of issues"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
