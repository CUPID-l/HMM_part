# Migration Guide - Upgrading to Enhanced Version

If you've been using the basic version of the HMM system, this guide will help you migrate to the enhanced version with Priority 1 improvements.

---

## What's New?

The enhanced version adds:
1. **Automated testing** (44 test cases)
2. **Model selection** (automatic K finding)
3. **Data validation** (quality checks)
4. **Config files** (YAML-based)

**Good news:** Your existing code will still work! These are **additions**, not breaking changes.

---

## Migration Steps

### Step 1: Update Dependencies

```powershell
# Backup your current environment
pip freeze > old_requirements.txt

# Install new dependencies
pip install pyyaml>=5.4.0
pip install pyarrow>=6.0.0
pip install hmmlearn>=0.2.7

# Verify installation
python -c "import yaml, pyarrow, hmmlearn; print('✅ All dependencies installed')"
```

### Step 2: Run Tests (Optional but Recommended)

```powershell
# Download test files if missing
# tests/test_features.py
# tests/test_model_wrapper.py
# tests/test_data_quality.py

# Run tests
python run_tests.py

# If tests pass, you're good to go!
```

### Step 3: Add Data Validation to Existing Code

**Before:**
```python
import pandas as pd
from src.features import compute_features

df = pd.read_parquet("data/raw/EUR_USD/latest.parquet")
features = compute_features(df)
```

**After (with validation):**
```python
import pandas as pd
from src.features import compute_features
from src.data_quality import validate_and_clean_data  # NEW

df = pd.read_parquet("data/raw/EUR_USD/latest.parquet")
df = validate_and_clean_data(df)  # NEW: Auto-clean
features = compute_features(df)
```

**Benefits:**
- Catches data issues before training
- Automatically removes bad rows
- Logs all data quality problems

---

### Step 4: (Optional) Convert to Config Files

If you have scripts with hard-coded parameters:

**Before:**
```python
# train_eurusd.py
from src.rolling_train import train_symbol

results = train_symbol(
    symbol='EUR_USD',
    window_size=252,
    step_size=21,
    n_states=3,
    max_iter=200,
    covariance_type='full'
)
```

**After (config-based):**
```python
# train_eurusd.py
from src.rolling_train import train_symbol
from src.config_manager import ConfigManager

config = ConfigManager('config/eurusd.yaml')
model_config = config.get_model_config()

results = train_symbol(
    symbol='EUR_USD',
    **model_config  # Uses all parameters from config
)
```

**Create config file:**
```yaml
# config/eurusd.yaml
model:
  n_states: 3
  window_size: 252
  step_size: 21
  max_iter: 200
  covariance_type: full
```

**Benefits:**
- Easy to change parameters
- Reproducible experiments
- Version control friendly

---

### Step 5: (Optional) Add Model Selection

If you're unsure about the optimal number of states:

**Before:**
```python
from src.model_wrapper import HMMRegimeModel

# Just guessing n_states=3
model = HMMRegimeModel(n_states=3)
model.fit(X)
```

**After (with auto-selection):**
```python
from src.model_selection import select_best_n_states

# Let the system decide
result = select_best_n_states(
    X,
    n_states_range=[2, 3, 4, 5],
    n_trials=5,
    criterion='bic'
)

model = result['best_model']
print(f"Selected {result['best_n_states']} states (BIC={result['best_score']:.2f})")
```

**Benefits:**
- Finds optimal number of states
- Avoids overfitting/underfitting
- Uses statistical criteria (BIC/AIC)

---

## Common Migration Patterns

### Pattern 1: Simple Script Enhancement

**Old script:**
```python
# old_train.py
from src.features import compute_features
from src.model_wrapper import HMMRegimeModel
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("data.parquet")
features = compute_features(df)

scaler = StandardScaler()
X = scaler.fit_transform(features)

model = HMMRegimeModel(n_states=3)
model.fit(X)
```

**Enhanced version:**
```python
# new_train.py
from src.features import compute_features
from src.model_wrapper import HMMRegimeModel
from src.data_quality import validate_and_clean_data  # NEW
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_parquet("data.parquet")
df = validate_and_clean_data(df)  # NEW: Validate first
features = compute_features(df)

scaler = StandardScaler()
X = scaler.fit_transform(features)

model = HMMRegimeModel(n_states=3)
model.fit(X)

# NEW: Run tests to verify
import subprocess
subprocess.run(['python', 'run_tests.py', '-q'])
```

### Pattern 2: Production Pipeline Enhancement

**Old pipeline:**
```python
# pipeline.py
import sys

symbols = ['EUR_USD', 'USD_JPY', 'GBP_USD']

for symbol in symbols:
    print(f"Training {symbol}...")
    # Hard-coded parameters
    os.system(f"python src/rolling_train.py --symbol {symbol} --window 252 --step 21 --n-states 3")
```

**Enhanced pipeline:**
```python
# enhanced_pipeline.py
from src.config_manager import ConfigManager
from src.rolling_train import train_symbol
from src.data_quality import validate_and_clean_data
import pandas as pd

# Load config once
config = ConfigManager('config/production.yaml')

for symbol in config.get('data.symbols'):
    print(f"Training {symbol}...")
    
    # Load and validate data
    data_file = f"data/raw/{symbol}/latest.parquet"
    df = pd.read_parquet(data_file)
    df = validate_and_clean_data(df)
    
    # Train with config parameters
    results = train_symbol(
        symbol=symbol,
        **config.get_model_config()
    )
    
    print(f"  ✅ Trained {results['n_windows']} windows")
```

### Pattern 3: Adding Tests to Existing Module

If you have custom functions, add tests:

**Your module:**
```python
# my_custom_module.py
def my_function(x):
    return x * 2
```

**Add tests:**
```python
# tests/test_my_custom_module.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from my_custom_module import my_function

def test_my_function():
    assert my_function(2) == 4
    assert my_function(0) == 0
    assert my_function(-1) == -2
```

**Run:**
```powershell
python run_tests.py tests/test_my_custom_module.py
```

---

## Backward Compatibility

✅ **All existing code continues to work**

The new features are **opt-in**:
- Don't want tests? Don't run them
- Don't want config files? Keep using command-line args
- Don't want model selection? Stick with fixed n_states
- Don't want validation? Skip the validate_and_clean_data call

**Nothing breaks!**

---

## Gradual Migration Strategy

You don't have to migrate everything at once. Suggested order:

### Week 1: Add Testing
- Run `python run_tests.py` to verify installation
- Optionally add tests for your custom code

### Week 2: Add Data Validation
- Add `validate_and_clean_data()` to your data loading code
- Monitor logs for data quality issues

### Week 3: Experiment with Model Selection
- Try `select_best_n_states()` on a few symbols
- Compare results with your fixed n_states

### Week 4: Migrate to Config Files
- Create config files for your different experiments
- Update scripts to use ConfigManager

---

## Troubleshooting Migration Issues

### Issue: Import errors

```python
ModuleNotFoundError: No module named 'yaml'
```

**Fix:**
```powershell
pip install pyyaml
```

### Issue: Tests fail

```
tests/test_features.py::test_rsi_range FAILED
```

**Fix:**
```powershell
# Check if you have correct pandas version
pip install --upgrade pandas scikit-learn

# Re-run tests
python run_tests.py -v
```

### Issue: Config file not found

```python
FileNotFoundError: config/default.yaml
```

**Fix:**
```powershell
# Create config directory
mkdir config

# Copy default config (if you have it)
Copy-Item path\to\default.yaml config\

# Or download from repository
```

---

## Verification Checklist

After migration, verify:

- [ ] All new dependencies installed (`pip list | grep -E "yaml|pyarrow|hmmlearn"`)
- [ ] Tests run successfully (`python run_tests.py`)
- [ ] Data validation works (`from src.data_quality import validate_and_clean_data`)
- [ ] Config loading works (`from src.config_manager import ConfigManager`)
- [ ] Model selection works (`from src.model_selection import select_best_n_states`)
- [ ] Your existing code still works (no breaking changes)

---

## Getting Help

If you encounter issues during migration:

1. **Check the docs:**
   - `PRIORITY1_COMPLETE.md` - Full documentation
   - `QUICKSTART_ENHANCED.md` - Quick examples

2. **Run diagnostics:**
   ```powershell
   python run_tests.py -v
   python src\config_manager.py --validate config\default.yaml
   ```

3. **Verify imports:**
   ```python
   python -c "from src.data_quality import DataQualityChecker; from src.model_selection import select_best_n_states; from src.config_manager import ConfigManager; print('✅ All imports OK')"
   ```

---

## Summary

**Migration is easy:**
1. Install 3 new packages
2. Optionally add data validation (1 line of code)
3. Optionally convert to config files
4. Optionally use model selection

**Your existing code continues to work unchanged.**

**Recommended timeline:** 1-4 weeks for gradual adoption

**Risk level:** Low (opt-in features, no breaking changes)

Happy migrating! 🚀
