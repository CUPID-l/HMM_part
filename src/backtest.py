"""
Backtesting framework for HMM-based regime strategies.

Implements regime-based trading strategies with comprehensive performance evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import argparse
import warnings
warnings.filterwarnings('ignore')

from utils import (
    setup_logging, load_model_artifact, calculate_performance_metrics,
    format_currency_pair
)
from features import compute_features
from inference import HMMInferenceEngine

logger = setup_logging()

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
    logger.info("vectorbt available for advanced backtesting")
except ImportError:
    VBT_AVAILABLE = False
    logger.info("vectorbt not available, using basic backtesting")


class RegimeStrategy:
    """Base class for regime-based trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(
        self, 
        regime_sequence: List[str], 
        features_df: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """
        Generate trading signals based on regime sequence.
        
        Args:
            regime_sequence: List of regime labels
            features_df: DataFrame with features and prices
            **kwargs: Strategy-specific parameters
            
        Returns:
            Series of trading signals (-1, 0, 1) indexed by date
        """
        raise NotImplementedError("Subclasses must implement generate_signals")


class SimpleRegimeStrategy(RegimeStrategy):
    """
    Simple regime-based strategy:
    - Bull regime: Long position
    - Bear regime: Short position (or flat)
    - Sideways regime: Flat position
    """
    
    def __init__(self, allow_short: bool = False):
        super().__init__("Simple Regime Strategy")
        self.allow_short = allow_short
    
    def generate_signals(
        self, 
        regime_sequence: List[str], 
        features_df: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """Generate simple regime signals."""
        signals = pd.Series(index=features_df.index, dtype=float)
        
        for i, (date, regime) in enumerate(zip(features_df.index, regime_sequence)):
            if regime == 'Bull':
                signals[date] = 1.0  # Long
            elif regime == 'Bear':
                signals[date] = -1.0 if self.allow_short else 0.0  # Short or flat
            else:  # Sideways or unknown
                signals[date] = 0.0  # Flat
        
        return signals


class MomentumRegimeStrategy(RegimeStrategy):
    """
    Momentum-enhanced regime strategy:
    - Bull regime: Long if momentum positive, else flat
    - Bear regime: Short if momentum negative, else flat
    - Sideways regime: Mean reversion signals
    """
    
    def __init__(self, momentum_lookback: int = 5, bb_threshold: float = 0.8):
        super().__init__("Momentum Regime Strategy")
        self.momentum_lookback = momentum_lookback
        self.bb_threshold = bb_threshold
    
    def generate_signals(
        self, 
        regime_sequence: List[str], 
        features_df: pd.DataFrame,
        **kwargs
    ) -> pd.Series:
        """Generate momentum-enhanced regime signals."""
        signals = pd.Series(index=features_df.index, dtype=float)
        
        # Compute momentum indicator
        momentum = features_df['log_return'].rolling(self.momentum_lookback).mean()
        
        # Bollinger Band position for mean reversion
        bb_position = kwargs.get('bb_position')
        if bb_position is None and 'Close' in kwargs:
            # Compute BB position if not provided
            close_prices = kwargs['Close']
            bb_ma = close_prices.rolling(20).mean()
            bb_std = close_prices.rolling(20).std()
            bb_upper = bb_ma + 2 * bb_std
            bb_lower = bb_ma - 2 * bb_std
            bb_position = (close_prices - bb_lower) / (bb_upper - bb_lower)
        
        for i, (date, regime) in enumerate(zip(features_df.index, regime_sequence)):
            if regime == 'Bull':
                # Long if positive momentum
                if pd.notna(momentum[date]) and momentum[date] > 0:
                    signals[date] = 1.0
                else:
                    signals[date] = 0.0
            
            elif regime == 'Bear':
                # Short if negative momentum
                if pd.notna(momentum[date]) and momentum[date] < 0:
                    signals[date] = -1.0
                else:
                    signals[date] = 0.0
            
            else:  # Sideways - mean reversion
                if bb_position is not None and pd.notna(bb_position[date]):
                    if bb_position[date] > self.bb_threshold:
                        signals[date] = -0.5  # Short when near upper band
                    elif bb_position[date] < (1 - self.bb_threshold):
                        signals[date] = 0.5   # Long when near lower band
                    else:
                        signals[date] = 0.0
                else:
                    signals[date] = 0.0
        
        return signals


class AdaptiveRegimeStrategy(RegimeStrategy):
    """
    Adaptive strategy that uses regime probabilities for position sizing.
    """
    
    def __init__(self, max_position: float = 1.0, confidence_threshold: float = 0.6):
        super().__init__("Adaptive Regime Strategy")
        self.max_position = max_position
        self.confidence_threshold = confidence_threshold
    
    def generate_signals(
        self, 
        regime_sequence: List[str], 
        features_df: pd.DataFrame,
        regime_probabilities: Optional[np.ndarray] = None,
        **kwargs
    ) -> pd.Series:
        """Generate adaptive signals based on regime confidence."""
        signals = pd.Series(index=features_df.index, dtype=float)
        
        for i, (date, regime) in enumerate(zip(features_df.index, regime_sequence)):
            # Get regime confidence if available
            confidence = 1.0  # Default confidence
            if regime_probabilities is not None and i < len(regime_probabilities):
                # Find the probability of the current regime
                regime_probs = regime_probabilities[i]
                if regime == 'Bull':
                    confidence = regime_probs[2] if len(regime_probs) > 2 else regime_probs[-1]
                elif regime == 'Bear':
                    confidence = regime_probs[0] if len(regime_probs) > 0 else regime_probs[0]
                else:  # Sideways
                    confidence = regime_probs[1] if len(regime_probs) > 1 else 0.5
            
            # Scale position by confidence
            if confidence >= self.confidence_threshold:
                if regime == 'Bull':
                    signals[date] = self.max_position * confidence
                elif regime == 'Bear':
                    signals[date] = -self.max_position * confidence
                else:
                    signals[date] = 0.0
            else:
                signals[date] = 0.0
        
        return signals


class HMMBacktester:
    """
    Comprehensive backtesting framework for HMM regime strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 10 bps
        slippage: float = 0.0005,         # 5 bps
        max_leverage: float = 1.0
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of price
            max_leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_leverage = max_leverage
        
        # Results storage
        self.backtest_results = {}
    
    def run_backtest(
        self,
        inference_engine: HMMInferenceEngine,
        ohlc_data: pd.DataFrame,
        strategy: RegimeStrategy,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
        rebalance_frequency: str = 'daily',
        **strategy_kwargs
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest of regime strategy.
        
        Args:
            inference_engine: Fitted HMM inference engine
            ohlc_data: OHLC price data
            strategy: Trading strategy instance
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency: How often to rebalance ('daily', 'weekly', etc.)
            **strategy_kwargs: Additional strategy parameters
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        logger.info(f"Running backtest with {strategy.name}")
        
        # Filter data by date range
        if start_date:
            ohlc_data = ohlc_data[ohlc_data.index >= start_date]
        if end_date:
            ohlc_data = ohlc_data[ohlc_data.index <= end_date]
        
        logger.info(f"Backtest period: {ohlc_data.index.min()} to {ohlc_data.index.max()}")
        
        # Get regime predictions for the full period
        regime_predictions = inference_engine.predict_regime(
            ohlc_data, 
            return_probabilities=True, 
            return_sequence=True
        )
        
        # Extract sequence data
        sequence_data = regime_predictions['sequence']
        regime_sequence = sequence_data['regimes']
        regime_dates = pd.to_datetime(sequence_data['dates'])
        regime_probabilities = np.array(sequence_data['posterior_probabilities'])
        
        # Compute features for strategy (may need additional indicators)
        features_df = compute_features(ohlc_data)
        
        # Align regime sequence with features
        aligned_regimes = []
        aligned_probs = []
        for date in features_df.index:
            # Find closest regime prediction
            idx = np.searchsorted(regime_dates, date)
            if idx > 0 and idx < len(regime_dates):
                # Use the most recent prediction
                aligned_regimes.append(regime_sequence[idx-1])
                aligned_probs.append(regime_probabilities[idx-1])
            elif len(regime_sequence) > 0:
                # Use first/last available
                aligned_regimes.append(regime_sequence[min(idx, len(regime_sequence)-1)])
                aligned_probs.append(regime_probabilities[min(idx, len(regime_probabilities)-1)])
            else:
                aligned_regimes.append('Sideways')  # Default
                aligned_probs.append(np.ones(inference_engine.n_states) / inference_engine.n_states)
        
        # Generate trading signals
        strategy_kwargs.update({
            'Close': ohlc_data['Close'],
            'regime_probabilities': np.array(aligned_probs)
        })
        
        signals = strategy.generate_signals(
            aligned_regimes, 
            features_df, 
            **strategy_kwargs
        )
        
        # Convert aligned_regimes list to pandas Series with proper index
        aligned_regimes_series = pd.Series(aligned_regimes, index=features_df.index)
        
        # Run portfolio simulation
        portfolio_results = self._simulate_portfolio(
            signals=signals,
            price_data=ohlc_data['Close'],
            regime_sequence=aligned_regimes_series
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_results,
            ohlc_data['Close']
        )
        
        # Regime-specific analysis
        regime_analysis = self._analyze_regime_performance(
            portfolio_results,
            aligned_regimes,
            signals
        )
        
        # Compile comprehensive results
        backtest_results = {
            'strategy': strategy.name,
            'period': {
                'start_date': ohlc_data.index.min().isoformat(),
                'end_date': ohlc_data.index.max().isoformat(),
                'n_days': len(ohlc_data)
            },
            'model_info': inference_engine.get_model_info(),
            'portfolio_results': portfolio_results,
            'performance_metrics': performance_metrics,
            'regime_analysis': regime_analysis,
            'regime_predictions': regime_predictions,
            'signals': signals.to_dict(),
            'parameters': {
                'initial_capital': self.initial_capital,
                'transaction_cost': self.transaction_cost,
                'slippage': self.slippage,
                'strategy_params': strategy_kwargs
            }
        }
        
        self.backtest_results[strategy.name] = backtest_results
        
        logger.info(f"Backtest complete. Total return: {performance_metrics['total_return']:.2%}, "
                   f"Sharpe: {performance_metrics['sharpe_ratio']:.3f}")
        
        return backtest_results
    
    def _simulate_portfolio(
        self,
        signals: pd.Series,
        price_data: pd.Series,
        regime_sequence: pd.Series
    ) -> Dict:
        """
        Simulate portfolio performance based on trading signals.
        
        This is a simplified event-driven backtest loop.
        """
        # --- FIX: Align all series to a common index before simulation ---
        # Use concat to align all three series at once
        aligned_df = pd.concat([price_data, signals, regime_sequence], axis=1, join='inner')
        aligned_df.columns = ['price', 'signal', 'regime']
        
        if aligned_df.empty:
            logger.warning("Price data and signals have no overlapping dates. Cannot simulate.")
            return {
                'returns': pd.Series(dtype=np.float64),
                'positions': pd.Series(dtype=np.float64),
                'trades': 0,
                'final_equity': self.initial_capital
            }
        
        # Use aligned data from here on
        price_data = aligned_df['price']
        signals = aligned_df['signal']
        # --- End of FIX ---

        # Initialize portfolio
        equity = pd.Series(index=price_data.index, dtype=np.float64)
        equity.iloc[0] = self.initial_capital
        positions = pd.Series(index=price_data.index, dtype=np.float64).fillna(0.0)
        transaction_costs = pd.Series(index=price_data.index, dtype=np.float64).fillna(0.0)
        trades_series = pd.Series(index=price_data.index, dtype=np.float64).fillna(0.0)
        current_position = 0.0
        trades = 0

        # Simulation loop
        for i in range(1, len(price_data)):
            prev_price = price_data.iloc[i-1]
            current_price = price_data.iloc[i]
            
            # Calculate equity change from price movement
            equity.iloc[i] = equity.iloc[i-1] + current_position * (current_price - prev_price)
            
            # --- FIX: Use previous day's signal to avoid lookahead bias ---
            target_signal = signals.iloc[i-1] if pd.notna(signals.iloc[i-1]) else 0.0
            
            # Execute trade if signal changes
            if target_signal != current_position:
                trades += 1
                
                # Calculate transaction cost
                trade_value = equity.iloc[i] * abs(target_signal - current_position)
                cost = trade_value * self.transaction_cost
                equity.iloc[i] -= cost
                
                # Record transaction cost and trade
                transaction_costs.iloc[i] = cost
                trades_series.iloc[i] = target_signal - current_position
                
                # Update position
                current_position = target_signal
            
            positions.iloc[i] = current_position
        
        # Final calculations
        returns = equity.pct_change().fillna(0.0)
        
        return {
            'returns': returns,
            'positions': positions,
            'portfolio_value': equity,
            'transaction_costs': transaction_costs,
            'trades': trades_series,
            'final_equity': equity.iloc[-1],
            'total_trades': trades
        }
    
    def _calculate_performance_metrics(
        self,
        portfolio_results: Dict[str, pd.Series],
        benchmark_prices: pd.Series
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        
        returns = portfolio_results['returns']
        portfolio_value = portfolio_results['portfolio_value']
        
        # Basic performance metrics
        basic_metrics = calculate_performance_metrics(returns)
        
        # Benchmark comparison (buy-and-hold)
        benchmark_returns = benchmark_prices.pct_change().fillna(0)
        benchmark_metrics = calculate_performance_metrics(benchmark_returns)
        
        # Additional metrics
        additional_metrics = {
            'final_portfolio_value': portfolio_value.iloc[-1],
            'total_transaction_costs': portfolio_results['transaction_costs'].sum(),
            'number_of_trades': (portfolio_results['trades'].abs() > 1e-8).sum(),
            'avg_trade_size': portfolio_results['trades'].abs().mean(),
            'benchmark_return': benchmark_metrics['total_return'],
            'benchmark_sharpe': benchmark_metrics['sharpe_ratio'],
            'excess_return': basic_metrics['total_return'] - benchmark_metrics['total_return'],
            'information_ratio': self._calculate_information_ratio(returns, benchmark_returns),
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0,
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if (returns < 0).any() else np.inf,
            'calmar_ratio': basic_metrics['cagr'] / abs(basic_metrics['max_drawdown']) if basic_metrics['max_drawdown'] < 0 else np.inf
        }
        
        return {**basic_metrics, **additional_metrics}
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        excess_returns = returns - benchmark_returns
        if len(excess_returns) > 1 and excess_returns.std() > 0:
            return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        return 0.0
    
    def _analyze_regime_performance(
        self,
        portfolio_results: Dict[str, pd.Series],
        regime_sequence: List[str],
        signals: pd.Series
    ) -> Dict[str, Any]:
        """Analyze performance by regime."""
        
        returns = portfolio_results['returns']
        
        regime_analysis = {}
        unique_regimes = list(set(regime_sequence))
        
        for regime in unique_regimes:
            # Get indices where we're in this regime
            regime_mask = pd.Series(regime_sequence, index=returns.index) == regime
            
            if regime_mask.sum() > 0:
                regime_returns = returns[regime_mask]
                regime_signals = signals[regime_mask]
                
                regime_stats = {
                    'frequency': regime_mask.mean(),
                    'avg_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'win_rate': (regime_returns > 0).mean(),
                    'avg_position': regime_signals.mean(),
                    'total_return': (1 + regime_returns).prod() - 1,
                    'n_periods': regime_mask.sum()
                }
                
                regime_analysis[regime] = regime_stats
        
        return regime_analysis
    
    def compare_strategies(
        self,
        strategies: List[RegimeStrategy],
        inference_engine: HMMInferenceEngine,
        ohlc_data: pd.DataFrame,
        **backtest_kwargs
    ) -> Dict[str, Any]:
        """Compare multiple strategies on the same data."""
        
        logger.info(f"Comparing {len(strategies)} strategies")
        
        strategy_results = {}
        
        for strategy in strategies:
            try:
                result = self.run_backtest(
                    inference_engine=inference_engine,
                    ohlc_data=ohlc_data,
                    strategy=strategy,
                    **backtest_kwargs
                )
                strategy_results[strategy.name] = result['performance_metrics']
                
            except Exception as e:
                logger.error(f"Failed to backtest {strategy.name}: {e}")
                strategy_results[strategy.name] = {'error': str(e)}
        
        # Create comparison summary
        comparison_metrics = [
            'total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 
            'information_ratio', 'win_rate', 'calmar_ratio'
        ]
        
        comparison_df = pd.DataFrame({
            strategy_name: {metric: results.get(metric, np.nan) 
                          for metric in comparison_metrics}
            for strategy_name, results in strategy_results.items()
            if 'error' not in results
        })
        
        # Rank strategies
        ranking_metrics = ['sharpe_ratio', 'calmar_ratio', 'information_ratio']
        rankings = {}
        for metric in ranking_metrics:
            if metric in comparison_df.index:
                rankings[metric] = comparison_df.loc[metric].rank(ascending=False).to_dict()
        
        comparison_summary = {
            'strategy_results': strategy_results,
            'comparison_table': comparison_df.to_dict(),
            'rankings': rankings,
            'best_strategy': {
                'by_sharpe': comparison_df.loc['sharpe_ratio'].idxmax() if 'sharpe_ratio' in comparison_df.index else None,
                'by_return': comparison_df.loc['total_return'].idxmax() if 'total_return' in comparison_df.index else None,
                'by_calmar': comparison_df.loc['calmar_ratio'].idxmax() if 'calmar_ratio' in comparison_df.index else None
            }
        }
        
        return comparison_summary


def run_backtest_from_config(
    model_path: str,
    data_file: str,
    strategy_config: Dict[str, Any],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run backtest from configuration files.
    
    Args:
        model_path: Path to trained HMM model
        data_file: Path to OHLC data file
        strategy_config: Strategy configuration dictionary
        output_file: Optional path to save results
        
    Returns:
        Backtest results
    """
    # Load inference engine
    engine = HMMInferenceEngine(model_path)
    
    # Load data
    if data_file.endswith('.parquet'):
        ohlc_data = pd.read_parquet(data_file)
    else:
        ohlc_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Create strategy
    strategy_type = strategy_config.get('type', 'simple')
    if strategy_type == 'simple':
        strategy = SimpleRegimeStrategy(
            allow_short=strategy_config.get('allow_short', False)
        )
    elif strategy_type == 'momentum':
        strategy = MomentumRegimeStrategy(
            momentum_lookback=strategy_config.get('momentum_lookback', 5),
            bb_threshold=strategy_config.get('bb_threshold', 0.8)
        )
    elif strategy_type == 'adaptive':
        strategy = AdaptiveRegimeStrategy(
            max_position=strategy_config.get('max_position', 1.0),
            confidence_threshold=strategy_config.get('confidence_threshold', 0.6)
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Initialize backtester
    backtester = HMMBacktester(
        initial_capital=strategy_config.get('initial_capital', 10000),
        transaction_cost=strategy_config.get('transaction_cost', 0.001),
        slippage=strategy_config.get('slippage', 0.0005)
    )
    
    # Run backtest
    results = backtester.run_backtest(
        inference_engine=engine,
        ohlc_data=ohlc_data,
        strategy=strategy,
        start_date=pd.to_datetime(strategy_config.get('start_date')) if strategy_config.get('start_date') else None,
        end_date=pd.to_datetime(strategy_config.get('end_date')) if strategy_config.get('end_date') else None
    )
    def convert_timestamps(obj):
        """Recursively convert Timestamps to strings"""
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, pd.Timestamp) else k: convert_timestamps(v) 
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_timestamps(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, (pd.Series, np.ndarray)):
            return obj.tolist()
        return obj
    results = convert_timestamps(results) # convert tiemestamps to strings for JSON serialization
    # Save results
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Backtest results saved to {output_file}")
    
    return results


def main():
    """Command line interface for backtesting."""
    parser = argparse.ArgumentParser(description="HMM regime strategy backtesting")
    
    parser.add_argument('--model-path', required=True, help='Path to trained HMM model')
    parser.add_argument('--data-file', required=True, help='Path to OHLC data file')
    parser.add_argument('--strategy', choices=['simple', 'momentum', 'adaptive'], 
                       default='simple', help='Strategy type')
    parser.add_argument('--output-file', help='Path to save results')
    
    # Strategy parameters
    parser.add_argument('--allow-short', action='store_true', 
                       help='Allow short positions (simple strategy)')
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Transaction cost (fraction)')
    parser.add_argument('--start-date', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Backtest end date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        # Build strategy configuration
        strategy_config = {
            'type': args.strategy,
            'allow_short': args.allow_short,
            'initial_capital': args.initial_capital,
            'transaction_cost': args.transaction_cost,
            'start_date': args.start_date,
            'end_date': args.end_date
        }
        
        # Run backtest
        results = run_backtest_from_config(
            model_path=args.model_path,
            data_file=args.data_file,
            strategy_config=strategy_config,
            output_file=args.output_file
        )
        
        # Print summary
        perf = results['performance_metrics']
        print(f"\nBacktest Results for {results['strategy']}")
        print(f"Period: {results['period']['start_date']} to {results['period']['end_date']}")
        print(f"Total Return: {perf['total_return']:.2%}")
        print(f"CAGR: {perf['cagr']:.2%}")
        print(f"Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {perf['max_drawdown']:.2%}")
        print(f"Win Rate: {perf['win_rate']:.2%}")
        print(f"Number of Trades: {perf['number_of_trades']}")
        
        # Regime analysis
        if 'regime_analysis' in results:
            print(f"\nRegime Analysis:")
            for regime, stats in results['regime_analysis'].items():
                print(f"  {regime}: Return={stats['avg_return']:.4f}, "
                     f"Frequency={stats['frequency']:.2%}, "
                     f"Sharpe={stats['sharpe']:.2f}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
