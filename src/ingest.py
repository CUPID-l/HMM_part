"""
Robust data ingestion module for Alpha Vantage API with rate limiting and error handling.
"""

import os
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
from dotenv import load_dotenv

from utils import setup_logging, ensure_directory_exists, format_currency_pair, validate_ohlc_data

# Load environment variables
load_dotenv()

logger = setup_logging()

class AlphaVantageIngester:
    """Alpha Vantage API client with robust error handling and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, base_pause: int = 12):
        """
        Initialize the ingester.
        
        Args:
            api_key: Alpha Vantage API key (will use env var if not provided)
            base_pause: Base pause between requests in seconds
        """
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided or set in ALPHA_VANTAGE_API_KEY env var")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.base_pause = base_pause
        self.session = requests.Session()
        
    def build_url(self, symbol: str, function: str = "FX_DAILY", outputsize: str = "full") -> str:
        """Build API URL for given parameters."""
        params = {
            'function': function,
            'apikey': self.api_key,
            'outputsize': outputsize
        }
        
        # Handle different symbol types
        if '/' in symbol or symbol.count('_') == 1:
            # Forex pair
            if '/' in symbol:
                from_symbol, to_symbol = symbol.split('/')
            else:
                from_symbol, to_symbol = symbol.split('_')
            params.update({
                'from_symbol': from_symbol,
                'to_symbol': to_symbol
            })
        else:
            # Stock or commodity
            params['symbol'] = symbol
            if function == "FX_DAILY":
                function = "TIME_SERIES_DAILY"
                params['function'] = function
        
        url = f"{self.base_url}?" + "&".join([f"{k}={v}" for k, v in params.items()])
        return url
    
    def make_request_with_retry(self, url: str, max_retries: int = 5) -> Dict:
        """Make API request with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Making API request (attempt {attempt + 1}/{max_retries})")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API-specific errors
                if "Error Message" in data:
                    raise ValueError(f"API Error: {data['Error Message']}")
                
                if "Note" in data:
                    # Rate limit hit
                    wait_time = (2 ** attempt) * self.base_pause
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                
                # Check if we got actual data
                time_series_keys = [
                    "Time Series FX (Daily)",
                    "Time Series (Daily)",
                    "Time Series (1min)",
                    "Time Series (5min)"
                ]
                
                if not any(key in data for key in time_series_keys):
                    logger.warning(f"No time series data found in response: {list(data.keys())}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt * 5)
                        continue
                    else:
                        raise ValueError("No time series data in API response")
                
                return data
                
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt * 5)
                else:
                    raise
        
        raise RuntimeError(f"Failed to fetch data after {max_retries} attempts")
    
    def parse_time_series_data(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse Alpha Vantage JSON response to DataFrame."""
        # Find the time series key
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            raise ValueError("No time series data found in response")
        
        time_series = data[time_series_key]
        
        # Convert to DataFrame
        df_data = []
        for date_str, values in time_series.items():
            try:
                # Handle different key formats
                open_key = next((k for k in values.keys() if 'open' in k.lower()), None)
                high_key = next((k for k in values.keys() if 'high' in k.lower()), None)
                low_key = next((k for k in values.keys() if 'low' in k.lower()), None)
                close_key = next((k for k in values.keys() if 'close' in k.lower()), None)
                
                if all([open_key, high_key, low_key, close_key]):
                    df_data.append({
                        'Date': pd.to_datetime(date_str),
                        'Open': float(values[open_key]),
                        'High': float(values[high_key]),
                        'Low': float(values[low_key]),
                        'Close': float(values[close_key]),
                        'Symbol': symbol
                    })
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid data point for {date_str}: {e}")
                continue
        
        if not df_data:
            raise ValueError("No valid data points found")
        
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Validate data
        if not validate_ohlc_data(df):
            logger.warning("OHLC data validation failed - some data may be inconsistent")
        
        return df
    
    def fetch_and_save(
        self, 
        symbol: str, 
        output_dir: str = "data/raw",
        function: str = "FX_DAILY"
    ) -> pd.DataFrame:
        """
        Fetch data from API and save both raw JSON and processed CSV/Parquet.
        
        Args:
            symbol: Trading symbol (e.g., 'EUR/USD', 'AAPL')
            output_dir: Directory to save data
            function: Alpha Vantage function to use
            
        Returns:
            DataFrame with OHLC data
        """
        logger.info(f"Fetching data for {symbol}")
        
        # Prepare directories
        output_path = Path(output_dir)
        symbol_formatted = format_currency_pair(symbol)
        symbol_dir = output_path / symbol_formatted
        ensure_directory_exists(str(symbol_dir / "dummy"))
        
        # Build URL and fetch data
        url = self.build_url(symbol, function)
        data = self.make_request_with_retry(url)
        
        # Save raw JSON with timestamp
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        raw_file = symbol_dir / f"{symbol_formatted}_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved raw data: {raw_file}")
        
        # Parse and save processed data
        df = self.parse_time_series_data(data, symbol)
        
        # Save as both CSV and Parquet
        csv_file = symbol_dir / f"{symbol_formatted}_{timestamp}.csv"
        parquet_file = symbol_dir / f"{symbol_formatted}_{timestamp}.parquet"
        
        df.to_csv(csv_file)
        df.to_parquet(parquet_file)
        
        logger.info(f"Saved processed data: {csv_file} and {parquet_file}")
        logger.info(f"Data shape: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
        
        # Add pause after successful request
        time.sleep(self.base_pause)
        
        return df
    
    def get_latest_data_file(self, symbol: str, output_dir: str = "data/raw") -> Optional[str]:
        """Get the most recent data file for a symbol."""
        symbol_formatted = format_currency_pair(symbol)
        symbol_dir = Path(output_dir) / symbol_formatted
        
        if not symbol_dir.exists():
            return None
        
        # Find most recent parquet file
        parquet_files = list(symbol_dir.glob(f"{symbol_formatted}_*.parquet"))
        if not parquet_files:
            return None
        
        latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
        return str(latest_file)
    
    def load_latest_data(self, symbol: str, output_dir: str = "data/raw") -> Optional[pd.DataFrame]:
        """Load the most recent data for a symbol."""
        latest_file = self.get_latest_data_file(symbol, output_dir)
        if latest_file:
            return pd.read_parquet(latest_file, parse_dates=['Date'] if 'Date' in pd.read_parquet(latest_file).columns else None)
        return None


def fetch_multiple_symbols(
    symbols: List[str], 
    output_dir: str = "data/raw",
    api_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple symbols with proper rate limiting.
    
    Args:
        symbols: List of trading symbols
        output_dir: Directory to save data
        api_key: Alpha Vantage API key
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    ingester = AlphaVantageIngester(api_key)
    results = {}
    
    logger.info(f"Fetching data for {len(symbols)} symbols")
    
    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
            df = ingester.fetch_and_save(symbol, output_dir)
            results[symbol] = df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            results[symbol] = None
    
    logger.info(f"Successfully fetched data for {sum(1 for v in results.values() if v is not None)} symbols")
    return results


def main():
    """Command line interface for data ingestion."""
    parser = argparse.ArgumentParser(description="Fetch market data from Alpha Vantage")
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=['EUR/USD', 'USD/JPY', 'GBP/USD'],
        help='Symbols to fetch (e.g., EUR/USD USD/JPY AAPL)'
    )
    parser.add_argument(
        '--output-dir', 
        default='data/raw',
        help='Output directory for data files'
    )
    parser.add_argument(
        '--api-key',
        help='Alpha Vantage API key (optional if set in environment)'
    )
    
    args = parser.parse_args()
    
    try:
        results = fetch_multiple_symbols(
            symbols=args.symbols,
            output_dir=args.output_dir,
            api_key=args.api_key
        )
        
        print(f"\nIngestion complete. Results:")
        for symbol, df in results.items():
            if df is not None:
                print(f"  {symbol}: {len(df)} records ({df.index.min()} to {df.index.max()})")
            else:
                print(f"  {symbol}: FAILED")
                
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()
