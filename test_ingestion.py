#!/usr/bin/env python3
"""
Simple test script to diagnose ingestion issues.
"""

import os
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_environment():
    """Test basic environment setup."""
    print("=== Environment Test ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check working directory
    print(f"Current directory: {os.getcwd()}")
    
    # Check .env file
    env_file = Path(".env")
    print(f".env file exists: {env_file.exists()}")
    if env_file.exists():
        print(f".env file size: {env_file.stat().st_size} bytes")
    
    print()

def test_imports():
    """Test all required imports."""
    print("=== Import Test ===")
    
    required_packages = [
        'pandas', 'numpy', 'requests', 'dotenv', 'pathlib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
    
    print()

def test_src_imports():
    """Test importing our source modules."""
    print("=== Source Module Import Test ===")
    
    try:
        from utils import setup_logging, ensure_directory_exists, format_currency_pair
        print("✓ utils module imported successfully")
        
        # Test logger
        logger = setup_logging()
        print("✓ Logger created successfully")
        
    except Exception as e:
        print(f"✗ utils import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from ingest import AlphaVantageIngester
        print("✓ ingest module imported successfully")
        
    except Exception as e:
        print(f"✗ ingest import failed: {e}")
        traceback.print_exc()
        return False
    
    print()
    return True

def test_api_key():
    """Test API key configuration."""
    print("=== API Key Test ===")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if api_key:
            print(f"✓ API key loaded: {api_key[:8]}...")
        else:
            print("✗ No API key found in environment")
            return False
            
    except Exception as e:
        print(f"✗ Error loading API key: {e}")
        return False
    
    print()
    return True

def test_ingester_creation():
    """Test creating the ingester object."""
    print("=== Ingester Creation Test ===")
    
    try:
        from ingest import AlphaVantageIngester
        ingester = AlphaVantageIngester()
        print(f"✓ Ingester created successfully")
        print(f"  Base URL: {ingester.base_url}")
        print(f"  Base pause: {ingester.base_pause}s")
        print(f"  API key set: {'Yes' if ingester.api_key else 'No'}")
        
    except Exception as e:
        print(f"✗ Failed to create ingester: {e}")
        traceback.print_exc()
        return False
    
    print()
    return True

def test_url_building():
    """Test URL building for different symbol types."""
    print("=== URL Building Test ===")
    
    try:
        from ingest import AlphaVantageIngester
        ingester = AlphaVantageIngester()
        
        # Test forex pair
        url1 = ingester.build_url("EUR/USD", "FX_DAILY")
        print(f"✓ Forex URL: {url1}")
        
        # Test stock symbol
        url2 = ingester.build_url("AAPL", "TIME_SERIES_DAILY")
        print(f"✓ Stock URL: {url2}")
        
    except Exception as e:
        print(f"✗ URL building failed: {e}")
        traceback.print_exc()
        return False
    
    print()
    return True

def test_simple_request():
    """Test making a simple API request."""
    print("=== Simple API Request Test ===")
    
    try:
        from ingest import AlphaVantageIngester
        ingester = AlphaVantageIngester()
        
        print("Making test request to Alpha Vantage API...")
        print("This may take a few seconds...")
        
        # Use a simple symbol
        url = ingester.build_url("EUR/USD", "FX_DAILY")
        print(f"Request URL: {url}")
        
        # Make the request
        data = ingester.make_request_with_retry(url, max_retries=2)
        
        print(f"✓ API request successful")
        print(f"  Response keys: {list(data.keys())}")
        
        # Check for time series data
        time_series_keys = [k for k in data.keys() if "Time Series" in k]
        if time_series_keys:
            ts_key = time_series_keys[0]
            ts_data = data[ts_key]
            print(f"  Time series key: {ts_key}")
            print(f"  Data points: {len(ts_data)}")
            
            # Show first date
            first_date = list(ts_data.keys())[0]
            print(f"  First date: {first_date}")
            print(f"  First data: {ts_data[first_date]}")
        else:
            print(f"  ⚠️  No time series data found")
            print(f"  Full response: {data}")
        
    except Exception as e:
        print(f"✗ API request failed: {e}")
        traceback.print_exc()
        return False
    
    print()
    return True

def main():
    """Run all diagnostic tests."""
    print("HMM Ingestion Diagnostic Tool")
    print("=" * 50)
    print()
    
    tests = [
        test_environment,
        test_imports,
        test_src_imports,
        test_api_key,
        test_ingester_creation,
        test_url_building,
        test_simple_request
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! The ingestion system should work properly.")
    else:
        print(f"\n❌ {sum(1 for r in results if not r)} tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
