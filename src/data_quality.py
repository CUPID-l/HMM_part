"""
Data quality validation utilities.

Provides comprehensive checks for OHLC data integrity.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Validate OHLC data quality before training"""
    
    @staticmethod
    def check_ohlc_consistency(df: pd.DataFrame) -> Dict:
        """
        Check if OHLC relationships are valid.
        
        Rules:
        - High >= Low (always)
        - High >= Open (always)
        - High >= Close (always)
        - Low <= Open (always)
        - Low <= Close (always)
        
        Args:
            df: DataFrame with OHLC columns
            
        Returns:
            Dict with 'valid' (bool) and 'issues' (list of strings)
        """
        issues = []
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {
                'valid': False,
                'issues': [f"Missing required columns: {missing_cols}"]
            }
        
        # High should be >= Low
        if (df['High'] < df['Low']).any():
            n_violations = (df['High'] < df['Low']).sum()
            issues.append(f"{n_violations} rows where High < Low")
        
        # High should be >= Open
        if (df['High'] < df['Open']).any():
            n_violations = (df['High'] < df['Open']).sum()
            issues.append(f"{n_violations} rows where High < Open")
        
        # High should be >= Close
        if (df['High'] < df['Close']).any():
            n_violations = (df['High'] < df['Close']).sum()
            issues.append(f"{n_violations} rows where High < Close")
        
        # Low should be <= Open
        if (df['Low'] > df['Open']).any():
            n_violations = (df['Low'] > df['Open']).sum()
            issues.append(f"{n_violations} rows where Low > Open")
        
        # Low should be <= Close
        if (df['Low'] > df['Close']).any():
            n_violations = (df['Low'] > df['Close']).sum()
            issues.append(f"{n_violations} rows where Low > Close")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    @staticmethod
    def check_missing_data(df: pd.DataFrame, max_gap_days: int = 5) -> Dict:
        """
        Check for missing dates and data gaps.
        
        Args:
            df: DataFrame with DatetimeIndex
            max_gap_days: Maximum acceptable gap between dates
            
        Returns:
            Dict with 'valid' (bool) and 'issues' (list of strings)
        """
        issues = []
        
        # Check if index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            return {
                'valid': False,
                'issues': ['Index is not DatetimeIndex']
            }
        
        # Check for NaN values
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            total_nans = df[nan_cols].isnull().sum().sum()
            issues.append(f"NaN values detected: {total_nans} total in columns {nan_cols}")
        
        # Check for date gaps
        if len(df) > 1:
            date_diffs = df.index.to_series().diff()
            large_gaps = date_diffs[date_diffs > pd.Timedelta(days=max_gap_days)]
            
            if len(large_gaps) > 0:
                max_gap = large_gaps.max().days
                issues.append(
                    f"{len(large_gaps)} gaps larger than {max_gap_days} days "
                    f"(largest gap: {max_gap} days)"
                )
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    @staticmethod
    def check_price_sanity(df: pd.DataFrame, max_daily_change: float = 0.20) -> Dict:
        """
        Check for unrealistic price movements.
        
        Args:
            df: DataFrame with OHLC columns
            max_daily_change: Maximum acceptable daily price change (default 20%)
            
        Returns:
            Dict with 'valid' (bool) and 'issues' (list of strings)
        """
        issues = []
        
        # Check for zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    n_violations = (df[col] <= 0).sum()
                    issues.append(f"{n_violations} zero or negative prices in {col}")
        
        # Check for extreme returns
        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change().dropna()
            extreme_returns = returns[abs(returns) > max_daily_change]
            
            if len(extreme_returns) > 0:
                max_return = abs(extreme_returns).max()
                issues.append(
                    f"{len(extreme_returns)} days with >{max_daily_change*100:.0f}% price change "
                    f"(max: {max_return*100:.1f}%)"
                )
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    @classmethod
    def validate_data(cls, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Run all validation checks on OHLC data.
        
        Args:
            df: DataFrame with OHLC data
            verbose: Whether to log issues
            
        Returns:
            Dict with validation results:
            {
                'all_valid': bool,
                'checks': {
                    'ohlc_consistency': {...},
                    'missing_data': {...},
                    'price_sanity': {...}
                }
            }
        """
        results = {
            'ohlc_consistency': cls.check_ohlc_consistency(df),
            'missing_data': cls.check_missing_data(df),
            'price_sanity': cls.check_price_sanity(df)
        }
        
        all_valid = all(r['valid'] for r in results.values())
        
        if verbose and not all_valid:
            logger.warning("Data quality issues detected:")
            for check_name, result in results.items():
                if not result['valid']:
                    logger.warning(f"  {check_name}:")
                    for issue in result['issues']:
                        logger.warning(f"    - {issue}")
        
        return {
            'all_valid': all_valid,
            'checks': results
        }


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data and apply automatic cleaning where possible.
    
    Args:
        df: Raw OHLC DataFrame
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        ValueError: If data has critical issues that can't be auto-fixed
    """
    checker = DataQualityChecker()
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Check for critical issues
    validation_result = checker.validate_data(df, verbose=True)
    
    # Auto-fix: Remove rows with NaN values
    if df.isnull().any().any():
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)
        logger.info(f"Removed {n_before - n_after} rows with NaN values")
    
    # Auto-fix: Ensure OHLC consistency
    ohlc_check = checker.check_ohlc_consistency(df)
    if not ohlc_check['valid']:
        logger.warning("Attempting to fix OHLC inconsistencies...")
        
        # Fix High: should be max of Open, High, Close, Low
        df['High'] = df[['Open', 'High', 'Close', 'Low']].max(axis=1)
        
        # Fix Low: should be min of Open, High, Close, Low
        df['Low'] = df[['Open', 'High', 'Close', 'Low']].min(axis=1)
        
        # Re-check
        ohlc_check = checker.check_ohlc_consistency(df)
        if not ohlc_check['valid']:
            raise ValueError(f"Could not fix OHLC inconsistencies: {ohlc_check['issues']}")
        
        logger.info("Fixed OHLC inconsistencies")
    
    # Check for critical issues that can't be auto-fixed
    price_check = checker.check_price_sanity(df)
    if not price_check['valid']:
        # Remove rows with zero or negative prices
        mask = (df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
        n_before = len(df)
        df = df[mask]
        n_removed = n_before - len(df)
        
        if n_removed > 0:
            logger.warning(f"Removed {n_removed} rows with zero/negative prices")
        
        # Re-check
        price_check = checker.check_price_sanity(df)
        if not price_check['valid']:
            # If still has issues, raise error
            raise ValueError(f"Data has critical price issues: {price_check['issues']}")
    
    # Final validation
    final_check = checker.validate_data(df, verbose=False)
    if not final_check['all_valid']:
        logger.warning(f"Data still has some issues after cleaning: {final_check}")
    
    return df
