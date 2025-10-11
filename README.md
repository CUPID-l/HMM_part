# Market Regime Detection using Hidden Markov Models

Complete end-to-end workflow for detecting market regimes using HMM on OHLC data.

## 📚 Documentation

This repository includes three comprehensive LaTeX reports:

1. **`combined_report.tex`** ⭐ **RECOMMENDED** - Complete report combining implemented HMM system with proposed deep learning extensions
   - **Part I:** Fully implemented HMM system with experimental results
   - **Part II:** Proposed BiLSTM-Attention architecture for future work
   - See [COMBINED_REPORT_GUIDE.md](COMBINED_REPORT_GUIDE.md) for details

2. **`report_hmm.tex`** - Focused report on the actual HMM implementation
   - Use for production system documentation
   - See [QUICK_START_REPORT.md](QUICK_START_REPORT.md) for details

3. **`main.tex`** - Aspirational hybrid system architecture (reference only)
   - See [COMPARISON_REPORT.md](COMPARISON_REPORT.md) for differences

## Features

- Raw OHLC data ingestion with rate limiting and robust error handling
- Feature engineering: log returns, volatility, RSI, MACD, Bollinger Band Width
- Rolling window HMM training with proper scaling and persistence
- Viterbi decoding and posterior probability computation
- Next-step regime probability prediction
- Regime-based backtesting framework
- Production-ready inference API

## Project Structure

```
market-regime-hmm/
├─ data/
│  ├─ raw/              # Raw CSVs from data sources
│  └─ processed/        # Processed feature data
├─ models/              # Saved models + scalers + metadata
├─ notebooks/           # Jupyter notebooks for EDA
├─ src/
│  ├─ ingest.py         # Data ingestion with rate limiting
│  ├─ features.py       # Feature engineering pipeline
│  ├─ model_wrapper.py  # HMM wrapper class
│  ├─ rolling_train.py  # Rolling window training
│  ├─ inference.py      # Production prediction API
│  ├─ backtest.py       # Backtesting framework
│  └─ utils.py          # Utility functions
├─ tests/               # Unit tests
├─ requirements.txt
└─ README.md
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file with your Alpha Vantage API key:
   ```
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```

3. **Fetch data:**
   ```bash
   python src/ingest.py --symbols EUR/USD USD/JPY XAUUSD
   ```

4. **Run rolling training:**
   ```bash
   python src/rolling_train.py --symbol EUR_USD --window 252 --step 21 --n-states 3
   ```

5. **Make predictions:**
   ```bash
   python src/inference.py --model models/latest_model.joblib --ohlc data/raw/new_data.csv
   ```

6. **Run backtest:**
   ```bash
   python src/backtest.py --model models/latest_model.joblib
   ```

## Reproducibility

- All random seeds are fixed and logged in model metadata
- Exact feature computation with specified parameters
- Standardized scaling per model window
- Complete model artifact versioning

## Model Artifacts

Each saved model includes:
- Trained HMM model
- StandardScaler fitted on training window
- Feature names and parameters
- State label mapping (Bull/Bear/Sideways)
- Training window dates and hyperparameters
- Performance metrics

## Next-Step Probabilities

The system computes next-step regime probabilities using two methods:
1. From most probable current state: P(S_{t+1}=j) = A[s*, j]
2. From posterior distribution: P(S_{t+1}=j) = (γ_t^T * A)_j

## Testing

Run unit tests:
```bash
pytest tests/
```
