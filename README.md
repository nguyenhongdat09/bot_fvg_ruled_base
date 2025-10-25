# FVG Trading Bot - Ruled Base

A comprehensive trading bot that uses Fair Value Gaps (FVG) detection combined with technical indicators for market analysis and backtesting.

## 📋 Features

### ✅ MT5 Data Download
- Download historical data from MetaTrader 5
- Support for multiple symbols (EURUSD, GBPUSD, USDJPY, etc.)
- Multiple timeframes (M1, M5, M15, M30, H1, H4, D1)
- Save/load data from CSV files
- Graceful fallback to sample data when MT5 is not available

### ✅ Technical Indicators
- **Trend Indicators**: EMA, SMA, ADX, Trend Detection
- **Momentum Indicators**: RSI, MACD, Stochastic, ROC, CCI
- **Volatility Indicators**: ATR, Bollinger Bands, Keltner Channels, Donchian Channels
- **Volume Indicators**: Volume MA, OBV, VWAP, MFI, Accumulation/Distribution

### ✅ FVG (Fair Value Gap) Detection
- Automatic FVG detection from OHLC data
- Bullish and Bearish FVG identification
- FVG strength calculation based on ATR
- FVG touch detection and tracking
- Market structure analysis
- 90-day lookback period

### ✅ Visualization
- Interactive charts with Plotly
- FVG zones visualization
- Indicator overlays
- Volume analysis
- Statistics and reports

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install pandas numpy plotly

# For MT5 data download (Windows only)
pip install MetaTrader5
```

### Installation

```bash
# Clone the repository
git clone https://github.com/nguyenhongdat09/bot_fvg_ruled_base.git
cd bot_fvg_ruled_base
```

### Configuration

Edit `config.py` to set your preferences:

```python
# MT5 Configuration (if using MT5)
MT5_CONFIG = {
    'login': YOUR_LOGIN,
    'password': 'YOUR_PASSWORD',
    'server': 'YOUR_SERVER',
}

# FVG Detection Parameters
FVG_CONFIG = {
    'lookback_days': 90,
    'min_gap_atr_ratio': 0.3,
    'atr_period': 14
}

# Indicator Parameters
INDICATOR_CONFIG = {
    'ema_periods': [20, 50, 200],
    'rsi_period': 14,
    'atr_period': 14,
    # ... more settings
}
```

## 📊 Usage

### 1. Test FVG + Indicators (Comprehensive Test)

This is the main test script that verifies all components are working:

```bash
python test_mt5_fvg_indicators.py
```

This script will:
1. Try to download data from MT5 (or use CSV/sample data)
2. Calculate all technical indicators
3. Detect FVGs
4. Create visualizations
5. Export results to CSV

**Output:**
- `logs/charts/mt5_fvg_indicators_chart.html` - Main chart with FVGs and indicators
- `logs/charts/fvg_statistics.html` - FVG statistics
- `logs/fvg_history.csv` - FVG history data

### 2. Test FVG Modules Only

To test only the FVG detection without MT5:

```bash
python test_fvg_complete.py
```

### 3. Download MT5 Data

To download data from MetaTrader 5:

```python
from data.download_mt5_data import initialize_mt5, download_ohlcv_data, save_data_to_csv

# Initialize MT5
initialize_mt5(login=YOUR_LOGIN, password=YOUR_PASSWORD, server=YOUR_SERVER)

# Download data
df = download_ohlcv_data('EURUSD', 'M15', num_bars=1000)

# Save to CSV
save_data_to_csv(df, 'EURUSD', 'M15')
```

### 4. Calculate Indicators

```python
from core.indicators.trend import calculate_ema, detect_trend_ema
from core.indicators.momentum import calculate_rsi, calculate_macd
from core.indicators.volatility import calculate_atr, calculate_bollinger_bands

# Calculate indicators
ema_20 = calculate_ema(data['close'], 20)
rsi = calculate_rsi(data['close'])
atr = calculate_atr(data['high'], data['low'], data['close'])
bb = calculate_bollinger_bands(data['close'])
```

### 5. Detect FVGs

```python
from core.fvg.fvg_manager import FVGManager

# Initialize FVG Manager
manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.3)

# Update with new candles
for i in range(20, len(data)):
    atr_value = data['atr'].iloc[i]
    manager.update(data.iloc[:i+1], i, atr_value)

# Get statistics
stats = manager.get_statistics()
print(f"Active FVGs: {stats['total_active']}")

# Get market structure
structure = manager.get_market_structure(current_price)
print(f"Bias: {structure['bias']}")
```

### 6. Create Visualizations

```python
from core.fvg.fvg_visualizer import FVGVisualizer

visualizer = FVGVisualizer(show_touched_fvgs=True, show_labels=True)

# Create FVG chart
visualizer.plot_fvg_chart(
    data,
    all_fvgs,
    title="FVG Analysis",
    show_volume=True,
    save_path='charts/fvg_chart.html'
)
```

## 📁 Project Structure

```
bot_fvg_ruled_base/
├── config.py                       # Configuration settings
├── main.py                         # Main entry point
├── test_mt5_fvg_indicators.py     # Comprehensive test script
├── test_fvg_complete.py           # FVG module tests
│
├── data/
│   ├── __init__.py
│   └── download_mt5_data.py       # MT5 data downloader
│
├── core/
│   ├── fvg/                       # FVG detection modules
│   │   ├── fvg_model.py          # FVG data model
│   │   ├── fvg_detector.py       # FVG detection logic
│   │   ├── fvg_manager.py        # FVG management & tracking
│   │   └── fvg_visualizer.py     # FVG visualization
│   │
│   ├── indicators/                # Technical indicators
│   │   ├── trend.py              # EMA, SMA, ADX
│   │   ├── momentum.py           # RSI, MACD, Stochastic
│   │   ├── volatility.py         # ATR, Bollinger Bands
│   │   └── volume.py             # Volume indicators
│   │
│   ├── strategy/                  # Trading strategies
│   │   └── signal_generator.py   # Signal generation
│   │
│   └── backtest/                  # Backtesting engine
│       └── backtester.py         # Backtest execution
│
└── logs/                          # Output files
    ├── charts/                    # HTML charts
    └── *.csv                      # Data exports
```

## 🔍 How It Works

### FVG Detection

Fair Value Gaps (FVG) are created by three consecutive candles where there's a gap between candle[i-2] and candle[i]:

- **Bullish FVG**: `high[i-2] < low[i]` - Gap below current price
- **Bearish FVG**: `low[i-2] > high[i]` - Gap above current price

### FVG Rules

1. **Gap Size**: Must be >= ATR × 0.3 (configurable)
2. **Touch = Invalidation**: When price touches FVG, it becomes inactive
3. **Lookback**: Only FVGs from last 90 days are tracked
4. **Strength**: Calculated as `gap_size / ATR`

### Market Structure Analysis

The FVG Manager analyzes market structure:

- **BULLISH_BIAS**: More bullish FVGs below price than bearish above
- **BEARISH_BIAS**: More bearish FVGs above price than bullish below
- **NO_FVG**: No active FVGs available

## 📈 Example Output

After running `test_mt5_fvg_indicators.py`:

```
✓ MT5 data download: READY
✓ Indicator calculations: READY
✓ FVG detection: READY
✓ Integration: WORKING

Results:
  - Total FVGs detected: 347
  - Active FVGs: 0
  - Bullish Touch Rate: 100.00%
  - Bearish Touch Rate: 100.00%

Indicators:
  - EMA-20: 1.10493
  - RSI: 55.87
  - ATR: 0.00084
  - BB Upper: 1.10587
```

## 🧪 Testing

Run all tests:

```bash
# Comprehensive test (MT5 + FVG + Indicators)
python test_mt5_fvg_indicators.py

# FVG modules only
python test_fvg_complete.py
```

## 📝 Status

✅ **READY FOR USE**

- [x] MT5 data download implementation
- [x] Technical indicators (Trend, Momentum, Volatility, Volume)
- [x] FVG detection and management
- [x] Visualization and reporting
- [x] Comprehensive testing
- [x] Integration working

## 🤝 Contributing

This is a private project. For questions, contact the repository owner.

## 📄 License

Private - All rights reserved.

## 🎯 Next Steps

1. Configure MT5 credentials in `config.py`
2. Run `test_mt5_fvg_indicators.py` to verify setup
3. Download real market data from MT5
4. Analyze FVGs and indicators
5. Develop trading strategies
6. Backtest strategies

## 💡 Tips

- **No MT5?** The system works with CSV files or sample data
- **Custom indicators?** Add them to `core/indicators/`
- **Different FVG rules?** Modify `FVG_CONFIG` in config.py
- **Want more visualization?** Check `core/fvg/fvg_visualizer.py`

---

**Project Status:** ✅ Ready to download data from MT5 and test FVG + indicators!
