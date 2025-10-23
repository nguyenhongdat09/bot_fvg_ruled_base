# FVG Module - Fair Value Gap Detection & Management

## Overview

Module này cung cấp đầy đủ chức năng phát hiện, quản lý và visualize Fair Value Gaps (FVG) trong trading.

## Components

### 1. FVG Model (`fvg_model.py`)

Định nghĩa FVG object với đầy đủ thuộc tính và methods.

```python
from core.fvg import FVG, generate_fvg_id, calculate_fvg_strength

# Create FVG
fvg = FVG(
    fvg_id=generate_fvg_id('BULLISH', timestamp, index),
    fvg_type='BULLISH',
    created_index=100,
    created_timestamp=timestamp,
    created_candle_indices=(98, 99, 100),
    top=1.10500,
    bottom=1.10300,
    strength=0.8,
    atr_at_creation=0.00025
)

# Check if touched
is_touched = fvg.check_touched(candle_high, candle_low, current_index, timestamp)

# Check if valid target
is_valid = fvg.is_valid_target(current_price)

# Get distance to price
distance = fvg.get_distance_to_price(current_price)

# Export to dict
fvg_dict = fvg.to_dict()
```

**Key Features:**
- ✅ FVG chạm = mất hiệu lực ngay lập tức
- ✅ Lookback 90 ngày
- ✅ Bullish FVG chạm khi: `candle_low <= FVG.top`
- ✅ Bearish FVG chạm khi: `candle_high >= FVG.bottom`

---

### 2. FVG Detector (`fvg_detector.py`)

Phát hiện FVG mới trong dữ liệu OHLC.

```python
from core.fvg import FVGDetector

# Initialize detector
detector = FVGDetector(
    min_gap_atr_ratio=0.3,  # Gap >= ATR × 0.3
    min_gap_pips=None       # Optional: minimum pips
)

# Detect FVG at specific index
fvg = detector.detect_fvg_at_index(data, index, atr)

# Detect all FVGs in data
fvgs = detector.detect_all_fvgs(data, atr_series, start_index=2)

# Get statistics
stats = detector.get_statistics(fvgs)
# Returns: {total, bullish, bearish, avg_gap_size, avg_strength}
```

**Detection Logic:**
- **Bullish FVG**: `High[i-2] < Low[i]` (gap ở giữa)
- **Bearish FVG**: `Low[i-2] > High[i]` (gap ở giữa)
- Lọc gap quá nhỏ bằng ATR ratio

---

### 3. FVG Manager (`fvg_manager.py`)

Quản lý toàn bộ FVG theo thời gian thực.

```python
from core.fvg import FVGManager, validate_signal_with_fvg, get_fvg_target

# Initialize manager
manager = FVGManager(
    lookback_days=90,
    min_gap_atr_ratio=0.3,
    min_gap_pips=None
)

# Update at each candle
for i in range(start_index, len(data)):
    new_fvg = manager.update(data.iloc[:i+1], i, atr.iloc[i])

    # Get market structure
    structure = manager.get_market_structure(data['close'].iloc[i])

    # Validate signal
    if validate_signal_with_fvg(structure, 'BUY'):
        target = get_fvg_target(structure, 'BUY')
        # Execute trade...

# Get statistics
stats = manager.get_statistics()

# Export data
history_df = manager.export_history_to_dataframe()
active_df = manager.export_active_to_dataframe()
```

**Market Structure (Bias):**

| Bias | FVG dưới | FVG trên | Action |
|------|----------|----------|--------|
| `BULLISH_BIAS` | ✅ | ❌ | Chỉ trade BUY |
| `BEARISH_BIAS` | ❌ | ✅ | Chỉ trade SELL |
| `BOTH_FVG` | ✅ | ✅ | Trade theo indicators |
| `NO_FVG` | ❌ | ❌ | ❌ NO TRADE |

---

### 4. FVG Visualizer (`fvg_visualizer.py`)

Tạo interactive charts với Plotly.

```python
from core.fvg import FVGVisualizer, quick_plot_fvgs

# Initialize visualizer
visualizer = FVGVisualizer(
    show_touched_fvgs=True,
    show_labels=True
)

# Plot main FVG chart
fig = visualizer.plot_fvg_chart(
    data=data,
    fvgs=fvgs,
    title="FVG Analysis",
    show_volume=True,
    signals=signals,  # Optional: trading signals
    save_path='logs/charts/fvg_chart.html'
)

# Plot statistics
fig_stats = visualizer.plot_fvg_statistics(
    fvgs=fvgs,
    save_path='logs/charts/statistics.html'
)

# Create full report
report_files = visualizer.create_fvg_report(
    data=data,
    fvgs=fvgs,
    signals=signals,
    output_dir='logs/charts'
)

# Quick plot helper
quick_plot_fvgs(data, fvgs, title="Quick View")
```

**Chart Features:**
- 📊 Interactive candlestick chart
- 🟢 Green zones: Bullish FVG active
- 🔴 Red zones: Bearish FVG active
- ⚫ Gray zones: FVG touched
- 📍 Signal markers (BUY/SELL)
- 📈 Volume subplot

---

## Complete Example

```python
import pandas as pd
from core.fvg import FVGManager, FVGVisualizer

# Load data
data = pd.read_csv('data/EURUSD_M15.csv', index_col='timestamp', parse_dates=True)

# Calculate ATR
def calculate_atr(data, period=14):
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

atr = calculate_atr(data)

# Initialize manager
manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.3)

# Process all candles
for i in range(20, len(data)):
    manager.update(data.iloc[:i+1], i, atr.iloc[i])

# Get final statistics
stats = manager.get_statistics()
print(f"Total FVGs created: {stats['total_bullish_created'] + stats['total_bearish_created']}")
print(f"Active FVGs: {stats['total_active']}")

# Get market structure
structure = manager.get_market_structure(data['close'].iloc[-1])
print(f"Market Bias: {structure['bias']}")

# Visualize
visualizer = FVGVisualizer()
visualizer.create_fvg_report(
    data=data,
    fvgs=manager.all_fvgs_history,
    output_dir='logs/charts'
)
```

---

## Testing

Chạy test script đầy đủ:

```bash
python test_fvg_complete.py
```

Test script sẽ:
1. ✅ Tạo sample data
2. ✅ Test FVG Model
3. ✅ Test FVG Detector
4. ✅ Test FVG Manager
5. ✅ Test FVG Visualizer
6. ✅ Export data to CSV/HTML

**Output files:**
- `logs/charts/test_fvg_chart.html` - Main chart
- `logs/charts/test_fvg_statistics.html` - Statistics
- `logs/fvg_history_test.csv` - FVG history
- `logs/fvg_active_test.csv` - Active FVGs

---

## Important Rules

### FVG Touched Logic

⚠️ **CRITICAL**: FVG chạm = mất hiệu lực ngay lập tức (KHÔNG có khái niệm lấp 50%)

- **Bullish FVG** bị chạm khi: `candle_low <= FVG.top`
- **Bearish FVG** bị chạm khi: `candle_high >= FVG.bottom`
- Khi touched: `is_active = False`, `is_touched = True`

### FVG Expiration

- FVG chỉ valid trong **90 ngày** (calendar days)
- Sau 90 ngày: tự động remove khỏi active list

### FVG Validation

FVG là target hợp lệ khi:
1. ✅ `is_active = True`
2. ✅ `is_touched = False`
3. ✅ Nằm đúng phía so với giá hiện tại:
   - Bullish FVG: Phải ở DƯỚI giá
   - Bearish FVG: Phải ở TRÊN giá

---

## Performance Tips

1. **Batch Processing**: Dùng `detect_all_fvgs()` thay vì loop `detect_fvg_at_index()`
2. **Lookback Limit**: Giảm `lookback_days` nếu không cần FVG cũ
3. **Visualization**: Set `show_touched_fvgs=False` để giảm clutter
4. **Export**: Export DataFrame chỉ khi cần (tốn memory)

---

## Roadmap

- [ ] Add FVG confluence scoring
- [ ] FVG mitigation zones (25%, 50%, 75%)
- [ ] FVG breakaway detection
- [ ] Multi-timeframe FVG alignment
- [ ] Machine learning FVG quality scoring

---

## Support

Nếu gặp vấn đề, check:
1. ATR values không None/NaN
2. Data có đủ OHLC columns
3. Index là DatetimeIndex
4. Plotly đã install: `pip install plotly`
