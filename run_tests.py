"""
Run all unit tests with coverage reporting.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py -k test_name # Run specific test
    python run_tests.py --cov        # With coverage report
"""

import sys
import pytest
from pathlib import Path

if __name__ == "__main__":
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Default arguments
    args = [
        "tests/",
        "-v",
        "--tb=short",
    ]
    
    # Add user arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Run pytest
    exit_code = pytest.main(args)
    sys.exit(exit_code)
