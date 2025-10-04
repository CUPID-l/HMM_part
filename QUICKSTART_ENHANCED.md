# Quick Start Guide - Enhanced Features

This guide shows you how to use the new Priority 1 improvements in your workflow.

---

## 1. Run Tests (Quality Assurance)

Before using the system, verify everything works:

```powershell
# Install test dependencies
pip install pytest pyyaml pyarrow hmmlearn

# Run all tests
python run_tests.py

# You should see: 44 passed
```

---

## 2. Use Configuration Files (Recommended)

Instead of passing command-line arguments, use a configuration file:

### Step 1: Create your config

```powershell
# Copy the default config
Copy-Item config\default.yaml config\my_config.yaml

# Edit it with your preferences
notepad config\my_config.yaml
```

### Step 2: Modify key settings

```yaml
# config/my_config.yaml

data:
  symbols:
    - EUR/USD
    - USD/JPY
  api_key: ${ALPHA_VANTAGE_API_KEY}

model:
  n_states: 3
  window_size: 252
  step_size: 21

model_selection:
  enabled: true  # <-- Enable automatic model selection
  n_states_range: [2, 3, 4]
  criterion: bic
```

### Step 3: Use the config in Python

```python
from src.config_manager import ConfigManager
from src.rolling_train import train_symbol

# Load config
config = ConfigManager('config/my_config.yaml')

# Train with config settings
model_config = config.get_model_config()

results = train_symbol(
    symbol='EUR_USD',
    window_size=model_config['window_size'],
    step_size=model_config['step_size'],
    n_states=model_config['n_states']
)
```

---

## 3. Validate Data Quality

Always validate your data before training:

```python
from src.data_quality import validate_and_clean_data
import pandas as pd

# Load raw data
df = pd.read_parquet("data/raw/EUR_USD/latest.parquet")

# Validate and auto-clean
clean_df = validate_and_clean_data(df)

print(f"Cleaned {len(df) - len(clean_df)} rows with issues")

# Now use clean_df for feature computation
```

---

## 4. Automatic Model Selection

Let the system find the best number of states:

```python
from src.model_selection import select_best_n_states
from src.features import compute_features
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and prepare data
df = pd.read_parquet("data/raw/EUR_USD/latest.parquet")
features = compute_features(df)

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(features.values)

# Automatic selection
result = select_best_n_states(
    X,
    n_states_range=[2, 3, 4, 5],  # Test these values
    n_trials=5,                   # 5 random initializations each
    criterion='bic'                # Use BIC for selection
)

print(f"Selected: {result['best_n_states']} states")
print(f"BIC: {result['best_score']:.2f}")

# Use the best model
best_model = result['best_model']
```

---

## 5. Complete Workflow Example

Putting it all together:

```python
"""
Complete workflow with all Priority 1 improvements
"""

from src.config_manager import ConfigManager
from src.data_quality import validate_and_clean_data
from src.features import compute_features
from src.model_selection import select_best_n_states
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# 1. Load configuration
config = ConfigManager('config/my_config.yaml')
print(f"Loaded config: {config.config_path}")

# 2. Load and validate data
symbol = 'EUR_USD'
data_file = f"data/raw/{symbol}/latest.parquet"
df = pd.read_parquet(data_file)
print(f"Loaded {len(df)} rows")

# Auto-clean data
df = validate_and_clean_data(df)
print(f"Cleaned data: {len(df)} rows")

# 3. Compute features
features = compute_features(df)
print(f"Computed {features.shape[1]} features")

# 4. Standardize
scaler = StandardScaler()
X = scaler.fit_transform(features.values)

# 5. Model selection (if enabled)
model_selection_config = config.get('model_selection', {})

if model_selection_config.get('enabled', False):
    print("Running automatic model selection...")
    result = select_best_n_states(
        X,
        n_states_range=model_selection_config['n_states_range'],
        n_trials=model_selection_config['n_trials'],
        criterion=model_selection_config['criterion']
    )
    best_model = result['best_model']
    n_states = result['best_n_states']
    print(f"Selected {n_states} states (BIC={result['best_score']:.2f})")
else:
    print("Using configured n_states...")
    from src.model_wrapper import HMMRegimeModel
    n_states = config.get('model.n_states')
    best_model = HMMRegimeModel(n_states=n_states, random_state=42)
    best_model.fit(X)
    print(f"Trained {n_states}-state model")

# 6. Save model with all metadata
model_artifact = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': list(features.columns),
    'config': config.to_dict(),
    'n_states': n_states,
    'symbol': symbol
}

output_path = f"models/{symbol}_{n_states}states_validated.joblib"
joblib.dump(model_artifact, output_path)
print(f"Saved model: {output_path}")

print("\n✅ Workflow complete!")
```

Save this as `train_with_validation.py` and run:

```powershell
python train_with_validation.py
```

---

## 6. Troubleshooting

### Test Failures

If tests fail:

```powershell
# Run with more detail
python run_tests.py -v --tb=long

# Run specific failing test
python run_tests.py -k test_name_here
```

### Configuration Errors

If config validation fails:

```powershell
# Validate your config
python src/config_manager.py --validate config/my_config.yaml

# Show config contents
python src/config_manager.py --show config/my_config.yaml
```

### Data Quality Issues

If data validation fails:

```python
from src.data_quality import DataQualityChecker

checker = DataQualityChecker()
result = checker.validate_data(df, verbose=True)

# Check each issue
for check_name, check_result in result['checks'].items():
    if not check_result['valid']:
        print(f"\n{check_name} issues:")
        for issue in check_result['issues']:
            print(f"  - {issue}")
```

---

## 7. Best Practices

### Always:
1. ✅ Run tests before deploying (`python run_tests.py`)
2. ✅ Validate data before training (`validate_and_clean_data`)
3. ✅ Use configuration files for reproducibility
4. ✅ Use model selection for new symbols/datasets

### Never:
1. ❌ Skip data validation
2. ❌ Train on data with NaN values
3. ❌ Use hard-coded parameters (use config files)
4. ❌ Ignore test failures

---

## 8. Next Steps

Now that you have these enhancements:

1. **Run the complete test suite** to verify installation
2. **Create a custom config file** for your symbols
3. **Validate your historical data**
4. **Run model selection** to find optimal parameters
5. **Deploy with confidence** knowing data is validated

For more details, see:
- `PRIORITY1_COMPLETE.md` - Full documentation
- `config/default.yaml` - Configuration template
- `tests/` - Test examples

---

Happy trading! 🚀
