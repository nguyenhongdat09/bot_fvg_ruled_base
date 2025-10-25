# Hướng Dẫn Backtest Chiến Lược FVG + Confluence

## Tổng Quan

Hệ thống backtest hoàn chỉnh để test chiến lược FVG + Confluence trên dữ liệu thật.

### Các Tính Năng

✅ **Virtual/Real Mode Switching**
- Virtual mode: Trade với lot size cố định
- Real mode: Tự động chuyển sau 3 lệnh thua liên tiếp
- Martingale: Lot size × 1.3 sau mỗi lệnh thua
- Reset về Virtual mode sau khi thắng

✅ **Position Sizing & Risk Management**
- ATR-based position sizing
- Dynamic SL/TP (SL = ATR × 1.5, TP = ATR × 3.0)
- Risk 2% mỗi trade
- Maximum lot size protection

✅ **Confluence Scoring System**
- FVG: 50% (primary signal)
- VWAP: 20% (price position)
- OBV: 15% (volume trend)
- Volume Spike: 15% (momentum)
- ADX Filter: Optional (trend strength)

✅ **Performance Metrics**
- Win Rate
- Profit Factor
- Maximum Drawdown
- Average Win/Loss
- Virtual vs Real mode breakdown
- Trade logging to CSV

---

## Cài Đặt

### 1. Download Data

```bash
# Chạy batch download để tải dữ liệu nhiều symbols
python data/batch_download_mt5_data.py
```

File config trong `config.py`:
```python
BATCH_DOWNLOAD_CONFIG = {
    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', ...],
    'timeframes': ['M15', 'H1', 'H4'],
    'days': 180,
}
```

### 2. Cấu Hình Backtest

Mở file `examples/run_backtest.py` và chỉnh:

```python
def main():
    # Default configuration
    SYMBOL = 'GBPUSD'         # Symbol để test
    TIMEFRAME = 'M15'          # Base timeframe
    DAYS = 180                 # Số ngày data
    FVG_TIMEFRAME = 'H1'       # FVG analysis timeframe

    INITIAL_BALANCE = 10000.0  # Số dư ban đầu
    RISK_PER_TRADE = 0.02      # 2% risk mỗi trade
    MIN_CONFIDENCE = 70.0      # Minimum confluence score (70%)
    ENABLE_ADX = True          # Enable ADX filter
```

### 3. Chạy Backtest

```bash
python examples/run_backtest.py
```

---

## Kết Quả Backtest

### Console Output

Backtest sẽ hiển thị:

1. **Initialization**: Setup strategy, indicators, backtester
2. **Trade Execution**: Real-time trade open/close với details
3. **Progress**: % hoàn thành
4. **Summary**: Performance metrics tổng hợp
5. **Additional Analysis**: Virtual vs Real mode breakdown

### Sample Output

```
====================================================================================================
🎯 Trade #280 OPENED - SELL
====================================================================================================
   Time: 2025-10-14 09:00:00
   Price: 1.33069
   Lot Size: 0.1 (VIRTUAL)
   SL: 1.33216
   TP: 1.32775
   Confluence Score: 100.0% (HIGH)
   FVG Bias: BEARISH_BIAS
   Balance: $10,268.95

🟢 Trade #280 CLOSED - WIN
   SELL @ 1.33069 -> 1.32775
   Exit: TP
   PnL: $28.70 (29.4 pips)
   Lot Size: 0.1 (VIRTUAL)
   Balance: $10,297.65
```

### Summary Metrics

```
================================================================================
BACKTEST SUMMARY
================================================================================

[TRADES] Trading Statistics:
   Total Trades: 281
   Winning Trades: 103
   Losing Trades: 178
   Win Rate: 36.65%

[PROFIT] Financial Results:
   Initial Balance: $10,000.00
   Final Balance: $10,278.65
   Total PnL: $278.65
   Return: 2.79%

[METRICS] Performance Metrics:
   Profit Factor: 1.08
   Max Drawdown: 3.49%
   Avg Win: $34.61 (30.0 pips)
   Avg Loss: $-18.46 (-15.9 pips)
   Largest Win: $179.34
   Largest Loss: $-84.13
================================================================================
```

### CSV Export

Kết quả được lưu tự động vào `data/backtest_[SYMBOL]_[TIMEFRAME]_[TIMESTAMP].csv`

Columns:
- `entry_time`, `entry_price`, `direction`
- `lot_size`, `sl_price`, `tp_price`
- `mode` (VIRTUAL/REAL)
- `exit_time`, `exit_price`, `exit_reason` (TP/SL/END)
- `pnl`, `pnl_pips`
- `confluence_score`, `confidence`, `fvg_bias`

---

## Tùy Chỉnh Chiến Lược

### 1. Thay Đổi Confluence Weights

File: `strategies/fvg_confluence_strategy.py`

```python
def _setup_confluence_scorer(self):
    # Thay đổi weights
    weights = {
        'fvg': 60,      # Tăng FVG weight
        'vwap': 15,     # Giảm VWAP weight
        'obv': 15,
        'volume': 10,
    }

    self.confluence_scorer = ConfluenceScorer(
        weights=weights,
        adx_enabled=True,
        adx_threshold=25.0  # Thay đổi ADX threshold
    )
```
# Lưu ý: TỔNG PHẢI = 100
    # 60 + 15 + 15 + 10 = 100 ✓
### 2. Thay Đổi SL/TP Multipliers

File: `core/backtest/backtester.py`

```python
@dataclass
class BacktestConfig:
    # Stop loss / Take profit
    atr_sl_multiplier: float = 2.0       # Thay từ 1.5 -> 2.0 (wider SL)
    atr_tp_multiplier: float = 4.0       # Thay từ 3.0 -> 4.0 (higher TP)
```

### 3. Thay Đổi Martingale Settings

File: `core/backtest/backtester.py`

```python
@dataclass
class BacktestConfig:
    # Martingale settings
    consecutive_losses_trigger: int = 5   # Thay từ 3 -> 5
    martingale_multiplier: float = 1.5    # Thay từ 1.3 -> 1.5
    max_lot_size: float = 5.0             # Thay từ 10.0 -> 5.0
```

### 4. Thay Đổi Min Confidence Score

File: `examples/run_backtest.py`

```python
MIN_CONFIDENCE = 80.0  # Tăng từ 70% lên 80% (more selective)
```

---

## Phân Tích Kết Quả

### 1. Win Rate Thấp (< 40%)

**Nguyên nhân có thể:**
- Confluence score threshold quá thấp (quá nhiều trades)
- FVG bias không đủ mạnh
- ADX filter không hoạt động đúng

**Giải pháp:**
- Tăng MIN_CONFIDENCE từ 70% lên 80%
- Enable ADX filter strict hơn (threshold = 30)
- Kiểm tra lại FVG detection parameters

### 2. Profit Factor Thấp (< 1.5)

**Nguyên nhân có thể:**
- Risk/Reward ratio không đủ (TP/SL ratio)
- Average win < 2× average loss

**Giải pháp:**
- Tăng TP multiplier (3.0 -> 4.0)
- Giảm SL multiplier nếu quá loose
- Thêm trailing stop

### 3. Max Drawdown Cao (> 10%)

**Nguyên nhân có thể:**
- Martingale too aggressive
- Consecutive losses quá nhiều

**Giải pháp:**
- Giảm martingale_multiplier (1.3 -> 1.2)
- Tăng consecutive_losses_trigger (3 -> 5)
- Giảm max_lot_size

### 4. Real Mode PnL < Virtual Mode PnL

**Nguyên nhân:**
- Martingale recovery không hiệu quả
- Consecutive losses pattern không phù hợp

**Giải pháp:**
- Disable martingale (set consecutive_losses_trigger = 999)
- Hoặc điều chỉnh multiplier nhỏ hơn

---

## Advanced Usage

### Chạy Backtest Nhiều Symbols

```python
symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

for symbol in symbols:
    print(f"\n{'='*100}")
    print(f"BACKTESTING {symbol}")
    print(f"{'='*100}")

    backtester, strategy, results = run_backtest(
        symbol=symbol,
        timeframe='M15',
        fvg_timeframe='H1',
        initial_balance=10000.0,
        min_confidence=70.0
    )

    # Compare results across symbols
```

### A/B Testing Strategies

```python
# Test with different confidence thresholds
for confidence in [60, 70, 80, 90]:
    backtester, _, _ = run_backtest(
        min_confidence=confidence
    )

    metrics = backtester.get_performance_metrics()
    print(f"Confidence {confidence}%: Win Rate = {metrics['win_rate']:.1f}%, Return = {metrics['return_pct']:.2f}%")
```

### Export to Excel for Analysis

```python
import pandas as pd

# Load backtest results
df = pd.read_csv('data/backtest_GBPUSD_M15_20251025_084719.csv')

# Analyze by time of day
df['hour'] = pd.to_datetime(df['entry_time']).dt.hour
hourly_stats = df.groupby('hour').agg({
    'pnl': ['sum', 'mean', 'count'],
    'exit_reason': lambda x: (x == 'TP').sum() / len(x) * 100
})

print(hourly_stats)

# Export to Excel
with pd.ExcelWriter('backtest_analysis.xlsx') as writer:
    df.to_excel(writer, sheet_name='All Trades', index=False)
    hourly_stats.to_excel(writer, sheet_name='Hourly Stats')
```

---

## Troubleshooting

### Error: "Data file not found"

```bash
# Download data trước
python data/batch_download_mt5_data.py
```

### Error: "No module named 'pandas_ta'"

Đã fix bằng cách dùng custom ATRIndicator. Nếu vẫn lỗi:
- Check file `core/fvg/multi_timeframe_manager.py` đã import ATRIndicator chưa
- Xóa bỏ mọi `import pandas_ta` trong code

### Backtest chạy chậm

- Giảm số ngày data (180 -> 90 days)
- Tăng MIN_CONFIDENCE để giảm số trades
- Disable progress printing

### Win Rate = 0%

- Kiểm tra confluence score có đạt threshold không
- Check FVG có detect được không (print fvg_structure)
- Verify ADX filter không quá strict

---

## Next Steps

### 1. Optimize Parameters
- Grid search cho best confluence weights
- Test nhiều timeframe combinations
- Find optimal SL/TP ratios

### 2. Add More Features
- Trailing stop
- Breakeven move
- Time-based filters (avoid news events)
- Correlation filter (avoid trading correlated pairs)

### 3. Walk-Forward Analysis
- Split data thành training/validation sets
- Optimize trên training, test trên validation
- Avoid overfitting

### 4. Live Trading Preparation
- Paper trading với live data
- Add slippage simulation
- Add spread costs
- Risk management rules

---

## Kết Luận

Hệ thống backtest đã hoàn chỉnh và sẵn sàng sử dụng. Kết quả hiện tại:
- ✅ Win Rate: 36.65% (cần cải thiện lên 45-50%)
- ✅ Profit Factor: 1.08 (cần cải thiện lên > 1.5)
- ✅ Return: 2.79% over 180 days (~5.6% annualized)

**Điểm mạnh:**
- Hệ thống chạy ổn định không crash
- Virtual/Real mode switching hoạt động
- Martingale recovery có effect
- Confluence scoring logic đúng

**Cần cải thiện:**
- Tăng win rate bằng cách selective hơn
- Improve risk/reward ratio
- Fine-tune indicator weights
- Add more filters

Bắt đầu bằng cách test nhiều symbols và timeframes khác nhau để tìm best combination!
