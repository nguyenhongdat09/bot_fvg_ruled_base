# HƯỚNG DẪN SỬ DỤNG MULTI-TIMEFRAME FVG

## 📖 Tổng quan

Document này hướng dẫn cách phân tích FVG và indicators trên nhiều khung thời gian khác nhau.

**Use case:**
- FVG phân tích trên H1 (hourly)
- RSI phân tích trên M15 (15-minute)
- Volume phân tích trên M5 (5-minute)
- Strategy trading trên M15

---

## 🎯 2 Phương pháp

### **Option 1: Manual Resample & Align**
- ✅ Đơn giản, dễ hiểu
- ✅ Dễ debug
- ⚠️ Code hơi dài

### **Option 2: MultiTimeframeManager** (KHUYẾN NGHỊ)
- ✅ Clean, professional
- ✅ Reusable
- ✅ Automatic resample & align

---

## 📝 Option 1: Manual Resample & Align

### Step-by-step guide:

```python
import pandas as pd
import pandas_ta as ta
from core.fvg.fvg_manager import FVGManager

# 1. Load M15 data (base timeframe)
m15_data = pd.read_csv('data/EURUSD_M15_30days.csv',
                       index_col=0, parse_dates=True)

# 2. Resample to H1
h1_data = m15_data.resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# 3. Calculate indicators
# H1 ATR for FVG
h1_data['atr'] = ta.atr(h1_data['high'], h1_data['low'],
                        h1_data['close'], length=14)

# M15 RSI
m15_data['rsi'] = ta.rsi(m15_data['close'], length=14)

# 4. Process H1 FVG sequentially (NO LOOK-AHEAD BIAS)
fvg_h1 = FVGManager()
h1_states = {}  # {timestamp: structure}

for i in range(20, len(h1_data)):
    # Sequential processing
    fvg_h1.update(h1_data.iloc[:i+1], i, h1_data.iloc[i]['atr'])
    structure = fvg_h1.get_market_structure(h1_data.iloc[i]['close'])

    h1_states[h1_data.index[i]] = {
        'bias': structure['bias'],
        'total_active_fvgs': structure['total_active_fvgs']
    }

# 5. Align H1 states to M15 (FORWARD FILL)
m15_data['h1_bias'] = None

for i in range(len(m15_data)):
    m15_time = m15_data.index[i]

    # Find latest H1 state BEFORE or AT m15_time
    valid_h1_times = [t for t in h1_states.keys() if t <= m15_time]

    if valid_h1_times:
        latest_h1_time = max(valid_h1_times)
        m15_data.loc[m15_time, 'h1_bias'] = h1_states[latest_h1_time]['bias']

# 6. Strategy on M15 with H1 FVG
for i in range(100, len(m15_data)):
    h1_bias = m15_data.iloc[i]['h1_bias']
    m15_rsi = m15_data.iloc[i]['rsi']

    # BUY when H1 bullish + M15 RSI oversold
    if h1_bias == 'BULLISH_BIAS' and m15_rsi < 30:
        signal = 'BUY'
```

---

## 🚀 Option 2: MultiTimeframeManager (KHUYẾN NGHỊ)

### Quick Start:

```python
import pandas as pd
import pandas_ta as ta
from core.fvg.multi_timeframe_manager import MultiTimeframeManager

# 1. Load M15 data
m15_data = pd.read_csv('data/EURUSD_M15_30days.csv',
                       index_col=0, parse_dates=True)

# 2. Initialize MultiTimeframeManager
mtf = MultiTimeframeManager(m15_data, base_timeframe='M15')

# 3. Add FVG timeframes
mtf.add_fvg_timeframe('H1')
mtf.add_fvg_timeframe('H4')
mtf.add_fvg_timeframe('D1')  # Daily

# 4. Calculate M15 indicators
m15_data['rsi'] = ta.rsi(m15_data['close'], length=14)
m15_data['volume_sma'] = ta.sma(m15_data['volume'], length=20)

# 5. Backtest strategy
for i in range(100, len(m15_data)):
    # Update all timeframes
    mtf.update(i)

    # Get FVG bias from multiple timeframes
    h1_bias = mtf.get_fvg_bias('H1', i)
    h4_bias = mtf.get_fvg_bias('H4', i)
    d1_bias = mtf.get_fvg_bias('D1', i)

    # Get M15 indicators
    m15_rsi = m15_data.iloc[i]['rsi']
    m15_volume = m15_data.iloc[i]['volume']
    m15_volume_sma = m15_data.iloc[i]['volume_sma']

    # Strategy: BUY when all timeframes bullish + RSI oversold
    if (h1_bias == 'BULLISH_BIAS' and
        h4_bias == 'BULLISH_BIAS' and
        d1_bias == 'BULLISH_BIAS' and
        m15_rsi < 30 and
        m15_volume > m15_volume_sma * 1.5):

        signal = 'BUY'
```

---

## 📊 API Reference: MultiTimeframeManager

### Constructor

```python
MultiTimeframeManager(base_data, base_timeframe='M15')
```

**Parameters:**
- `base_data`: DataFrame OHLCV (smallest timeframe)
- `base_timeframe`: Base timeframe name (e.g., 'M15')

### Methods

#### `add_fvg_timeframe(timeframe, lookback_days=90, min_gap_atr_ratio=0.3)`

Add FVG analysis for a timeframe.

**Supported timeframes:**
- `'M1'`: 1-minute
- `'M5'`: 5-minute
- `'M15'`: 15-minute
- `'M30'`: 30-minute
- `'H1'`: 1-hour
- `'H4'`: 4-hour
- `'D1'`: Daily
- `'W1'`: Weekly
- `'MN1'`: Monthly

**Example:**
```python
mtf.add_fvg_timeframe('H1', lookback_days=90)
mtf.add_fvg_timeframe('H4', lookback_days=180)
```

#### `update(base_index)`

Update all timeframes up to base_index.

**Example:**
```python
for i in range(100, len(base_data)):
    mtf.update(i)
```

#### `get_fvg_bias(timeframe, base_index) -> str`

Get FVG bias at base_index.

**Returns:** `'BULLISH_BIAS'`, `'BEARISH_BIAS'`, `'BOTH_FVG'`, `'NO_FVG'`, or `None`

**Example:**
```python
h1_bias = mtf.get_fvg_bias('H1', i)
```

#### `get_fvg_structure(timeframe, base_index) -> dict`

Get full FVG structure at base_index.

**Returns:**
```python
{
    'timestamp': pd.Timestamp,
    'bias': str,
    'total_active_fvgs': int,
    'active_bullish': int,
    'active_bearish': int,
    'nearest_bullish_target': FVG or None,
    'nearest_bearish_target': FVG or None
}
```

**Example:**
```python
h1_structure = mtf.get_fvg_structure('H1', i)
print(f"H1 bias: {h1_structure['bias']}")
print(f"Active FVGs: {h1_structure['total_active_fvgs']}")
```

#### `get_statistics(timeframe) -> dict`

Get FVG statistics for a timeframe.

**Example:**
```python
h1_stats = mtf.get_statistics('H1')
print(f"Total created: {h1_stats['total_bullish_created']}")
print(f"Touch rate: {h1_stats['bullish_touch_rate']}%")
```

#### `get_available_timeframes() -> List[str]`

Get list of added timeframes.

**Example:**
```python
timeframes = mtf.get_available_timeframes()
# ['H1', 'H4', 'D1']
```

---

## 🧪 Verify No Look-Ahead Bias

### Test method:

```python
# Test 1: Backtest với 1000 candles
m15_test1 = m15_data.iloc[:1000].copy()
mtf1 = MultiTimeframeManager(m15_test1, 'M15')
mtf1.add_fvg_timeframe('H1')

signals1 = []
for i in range(100, len(m15_test1)):
    mtf1.update(i)
    h1_bias = mtf1.get_fvg_bias('H1', i)
    if h1_bias == 'BULLISH_BIAS':
        signals1.append(i)

# Test 2: Backtest với 1500 candles (nhưng chỉ lấy first 1000 signals)
m15_test2 = m15_data.iloc[:1500].copy()
mtf2 = MultiTimeframeManager(m15_test2, 'M15')
mtf2.add_fvg_timeframe('H1')

signals2 = []
for i in range(100, 1000):  # Chỉ check first 1000
    mtf2.update(i)
    h1_bias = mtf2.get_fvg_bias('H1', i)
    if h1_bias == 'BULLISH_BIAS':
        signals2.append(i)

# Kết quả PHẢI GIỐNG NHAU!
assert signals1 == signals2  # ✅ No look-ahead bias
```

---

## ⚠️ Common Mistakes (TRÁNH)

### ❌ **Mistake 1: Using method='nearest' instead of 'ffill'**

```python
# SAI - có thể lấy future data!
tf_idx = resampled.index.get_indexer([base_time], method='nearest')[0]

# ĐÚNG - chỉ lấy past data
tf_idx = resampled.index.get_indexer([base_time], method='ffill')[0]
```

### ❌ **Mistake 2: Not processing sequentially**

```python
# SAI - update toàn bộ trước
for i in range(len(h1_data)):
    fvg_h1.update(h1_data.iloc[:i+1], i, h1_data.iloc[i]['atr'])

# Sau đó mới iterate M15
for i in range(len(m15_data)):
    # Lấy H1 bias tại i
    ...

# ĐÚNG - iterate M15, update H1 động
for i in range(len(m15_data)):
    mtf.update(i)  # Tự động update H1 nếu cần
    h1_bias = mtf.get_fvg_bias('H1', i)
```

### ❌ **Mistake 3: Using future candle close**

```python
# SAI - dùng H1 candle chưa đóng
h1_current = h1_data.iloc[h1_idx]  # H1 candle có thể chưa close!

# ĐÚNG - chỉ dùng H1 candle đã đóng
h1_idx = resampled.index.get_indexer([base_time], method='ffill')[0]
```

---

## 📈 Strategy Examples

### Example 1: Triple Timeframe Confluence

```python
# H4 trend, H1 setup, M15 entry
mtf = MultiTimeframeManager(m15_data, 'M15')
mtf.add_fvg_timeframe('H4')
mtf.add_fvg_timeframe('H1')

for i in range(100, len(m15_data)):
    mtf.update(i)

    h4_bias = mtf.get_fvg_bias('H4', i)  # Trend
    h1_bias = mtf.get_fvg_bias('H1', i)  # Setup
    m15_rsi = m15_data.iloc[i]['rsi']     # Entry

    # BUY: H4 + H1 bullish + M15 oversold
    if h4_bias == 'BULLISH_BIAS' and h1_bias == 'BULLISH_BIAS' and m15_rsi < 30:
        signal = 'BUY'
```

### Example 2: Divergence Detection

```python
# H1 bullish but M15 bearish -> wait for correction
for i in range(100, len(m15_data)):
    mtf.update(i)

    h1_bias = mtf.get_fvg_bias('H1', i)
    h1_structure = mtf.get_fvg_structure('H1', i)

    # Get local M15 FVG
    m15_manager = FVGManager()
    m15_manager.update(m15_data.iloc[:i+1], i, m15_data.iloc[i]['atr'])
    m15_structure = m15_manager.get_market_structure(m15_data.iloc[i]['close'])

    # Divergence: H1 bullish but M15 bearish
    if h1_bias == 'BULLISH_BIAS' and m15_structure['bias'] == 'BEARISH_BIAS':
        # Wait for M15 to align with H1
        wait_for_alignment = True
```

---

## 🎓 Best Practices

### 1. Choose Base Timeframe Wisely
- Base timeframe = khung nhỏ nhất bạn cần
- Nếu trade M15 → base = M15
- Nếu trade M5 → base = M5

### 2. Limit Number of Timeframes
- Không nên dùng quá 3-4 timeframes
- Recommended: H4 (trend) + H1 (setup) + M15 (entry)

### 3. Always Verify No Look-Ahead Bias
- Test với data khác nhau
- Kết quả phải consistent

### 4. Use MultiTimeframeManager for Production
- Clean code
- Easier to maintain
- Professional

---

## 📁 Files Reference

- `core/fvg/multi_timeframe_manager.py` - MultiTimeframeManager class
- `examples/multi_timeframe_example.py` - Full working examples
- `MULTI_TIMEFRAME_ANALYSIS.md` - Chi tiết phân tích và so sánh
- `FVG_NO_LOOK_AHEAD_ANALYSIS.md` - Verify no look-ahead bias

---

## 🚀 Quick Start

```bash
# Run example
python examples/multi_timeframe_example.py
```

---

**CREATED:** 2025-10-24
**STATUS:** ✅ READY TO USE
