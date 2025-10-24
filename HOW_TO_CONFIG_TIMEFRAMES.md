# HƯỚNG DẪN CONFIG TIMEFRAMES

## 📖 Tổng quan

Sau này khi test strategy, bạn chỉ cần vào **1 FILE DUY NHẤT** để chỉnh timeframe cho tất cả components:

**FILE:** `config.py`

**SECTION:** `MULTI_TIMEFRAME_STRATEGY_CONFIG`

---

## 🎯 Cách Config Timeframe

### **Bước 1: Mở file `config.py`**

```bash
# Windows
notepad config.py

# Linux/Mac
nano config.py
# hoặc
code config.py  # VS Code
```

---

### **Bước 2: Tìm section `MULTI_TIMEFRAME_STRATEGY_CONFIG`**

Kéo xuống dòng ~189, bạn sẽ thấy:

```python
# ============================================
# MULTI-TIMEFRAME STRATEGY CONFIGURATION
# ============================================
MULTI_TIMEFRAME_STRATEGY_CONFIG = {
    # Base timeframe (smallest timeframe for execution)
    'base_timeframe': 'M15',       # Trading execution timeframe

    # FVG Analysis Timeframes
    'fvg_timeframes': {
        'primary': 'H1',           # Primary FVG timeframe
        'secondary': 'H4',         # Secondary FVG timeframe (optional)
        'tertiary': None,          # Tertiary FVG timeframe (optional)
    },

    # Indicators Timeframes
    'indicator_timeframes': {
        # Trend indicators
        'ema_fast': 'M15',         # Fast EMA timeframe
        'ema_slow': 'H1',          # Slow EMA timeframe
        'macd': 'M15',             # MACD timeframe

        # Momentum indicators
        'rsi': 'M15',              # RSI timeframe
        'stochastic': 'M15',       # Stochastic timeframe

        # Volatility indicators
        'atr': 'M15',              # ATR timeframe
        'bollinger': 'H1',         # Bollinger Bands timeframe

        # Volume indicators
        'volume_sma': 'M15',       # Volume SMA timeframe
        'obv': 'H1',               # OBV timeframe
        'cmf': 'M15',              # CMF timeframe
    },
    ...
}
```

---

### **Bước 3: Chỉnh Timeframe theo ý muốn**

**Ví dụ bạn muốn:**
- FVG phân tích trên **H1**
- RSI phân tích trên **M15**
- MACD phân tích trên **H1**
- Volume trên **M5**

**Chỉnh như sau:**

```python
MULTI_TIMEFRAME_STRATEGY_CONFIG = {
    'base_timeframe': 'M15',       # Execution timeframe (không đổi)

    'fvg_timeframes': {
        'primary': 'H1',           # ✅ FVG trên H1
        'secondary': 'H4',         # Optional: thêm H4 để confirm
        'tertiary': None,          # Không dùng
    },

    'indicator_timeframes': {
        'macd': 'H1',              # ✅ MACD trên H1 (đổi từ M15)
        'rsi': 'M15',              # ✅ RSI trên M15
        'volume_sma': 'M5',        # ✅ Volume trên M5 (đổi từ M15)

        # Các indicators khác giữ nguyên hoặc đổi theo ý muốn
        'atr': 'M15',
        'bollinger': 'H1',
        ...
    },
}
```

---

### **Bước 4: Save file**

```bash
# Ctrl+S (Windows/Linux)
# Cmd+S (Mac)
```

---

### **Bước 5: Chạy strategy**

```bash
python strategies/multi_timeframe_strategy.py
```

**Strategy sẽ TỰ ĐỘNG:**
- Setup FVG manager với timeframe bạn config
- Calculate indicators trên timeframe bạn chọn
- Align tất cả về base timeframe
- Backtest strategy

---

## 📋 Supported Timeframes

| Code | Timeframe | Mô tả |
|------|-----------|-------|
| `M1` | 1-minute | 1 phút |
| `M5` | 5-minute | 5 phút |
| `M15` | 15-minute | 15 phút |
| `M30` | 30-minute | 30 phút |
| `H1` | 1-hour | 1 giờ |
| `H4` | 4-hour | 4 giờ |
| `D1` | Daily | 1 ngày |
| `W1` | Weekly | 1 tuần |
| `MN1` | Monthly | 1 tháng |

---

## 🎯 Use Cases

### **Use Case 1: FVG H1 + RSI M15 + Volume M15**

```python
MULTI_TIMEFRAME_STRATEGY_CONFIG = {
    'base_timeframe': 'M15',

    'fvg_timeframes': {
        'primary': 'H1',           # FVG H1
        'secondary': None,
        'tertiary': None,
    },

    'indicator_timeframes': {
        'rsi': 'M15',              # RSI M15
        'volume_sma': 'M15',       # Volume M15
        'macd': 'M15',
        'atr': 'M15',
    },
}
```

---

### **Use Case 2: Multi-timeframe Confluence (H4 trend + H1 setup + M15 entry)**

```python
MULTI_TIMEFRAME_STRATEGY_CONFIG = {
    'base_timeframe': 'M15',       # Entry timeframe

    'fvg_timeframes': {
        'primary': 'H1',           # Setup FVG
        'secondary': 'H4',         # Trend FVG
        'tertiary': 'D1',          # Long-term bias
    },

    'indicator_timeframes': {
        'rsi': 'M15',              # Entry timing
        'macd': 'H1',              # Setup confirmation
        'ema_slow': 'H4',          # Trend direction
        'volume_sma': 'M15',
        'atr': 'M15',
    },
}
```

---

### **Use Case 3: Mỗi indicator một timeframe khác nhau**

```python
MULTI_TIMEFRAME_STRATEGY_CONFIG = {
    'base_timeframe': 'M15',

    'fvg_timeframes': {
        'primary': 'H1',
        'secondary': 'H4',
        'tertiary': None,
    },

    'indicator_timeframes': {
        # Trend
        'ema_fast': 'M15',         # Fast trend M15
        'ema_slow': 'H1',          # Slow trend H1
        'macd': 'H1',              # MACD H1

        # Momentum
        'rsi': 'M15',              # RSI M15
        'stochastic': 'M5',        # Stochastic M5

        # Volume
        'volume_sma': 'M5',        # Volume M5
        'obv': 'H1',               # OBV H1

        # Volatility
        'atr': 'M15',              # ATR M15
        'bollinger': 'H1',         # Bollinger H1
    },
}
```

---

## ⚠️ Lưu ý quan trọng

### **1. Base Timeframe**

- `base_timeframe` là khung nhỏ nhất (execution timeframe)
- Thường là M15 hoặc M5
- **TẤT CẢ indicators/FVG sẽ align về base timeframe này**

**Ví dụ:**
```python
'base_timeframe': 'M15',  # Execute trade trên M15
'fvg_timeframes': {
    'primary': 'H1',      # FVG H1 sẽ align xuống M15
}
```

Khi iterate qua M15:
- M15 index 100 = 10:00
- H1 FVG state tại 10:00 được lấy từ H1 candle gần nhất (forward fill)

---

### **2. Timeframe Hierarchy**

**ĐÚNG:** Timeframe lớn hơn >= Base timeframe
```python
'base_timeframe': 'M15',
'fvg_timeframes': {
    'primary': 'H1',       # ✅ H1 > M15
    'secondary': 'H4',     # ✅ H4 > M15
}
```

**SAI:** Timeframe nhỏ hơn < Base timeframe
```python
'base_timeframe': 'M15',
'fvg_timeframes': {
    'primary': 'M5',       # ❌ M5 < M15 (không hợp lý!)
}
```

**Lý do:** Không thể downsample M15 xuống M5. Chỉ có thể upsample M15 lên H1, H4, D1...

---

### **3. None = Không sử dụng**

Nếu không muốn dùng indicator/FVG nào, set = `None`:

```python
'fvg_timeframes': {
    'primary': 'H1',
    'secondary': None,     # ✅ Không dùng secondary FVG
    'tertiary': None,      # ✅ Không dùng tertiary FVG
}
```

---

### **4. Data Requirements**

Đảm bảo có đủ data cho base timeframe:

```bash
# Download M15 data
python data/download_mt5_data.py
```

File sẽ tự động tạo: `data/EURUSD_M15_30days.csv`

Strategy sẽ tự động resample M15 → H1, H4, D1...

---

## 🧪 Test Config

### **Bước 1: Chỉnh config**

```python
# config.py
MULTI_TIMEFRAME_STRATEGY_CONFIG = {
    'base_timeframe': 'M15',
    'fvg_timeframes': {
        'primary': 'H1',      # Bạn muốn test FVG H1
    },
    'indicator_timeframes': {
        'rsi': 'M15',         # RSI M15
        'macd': 'H1',         # MACD H1
    },
}
```

---

### **Bước 2: Chạy strategy**

```bash
python strategies/multi_timeframe_strategy.py
```

---

### **Bước 3: Xem output**

```
================================
MULTI-TIMEFRAME STRATEGY INITIALIZATION
================================
Base Timeframe: M15
Base Data: 2000 candles

📊 Setting up FVG Manager...
  Adding PRIMARY FVG timeframe: H1

📈 Calculating Indicators...
  RSI on M15
  MACD on H1
  ATR on M15
  Volume SMA on M15

================================
RUNNING BACKTEST
================================

BUY Signal at 2024-01-15 10:30:00
  Price: 1.09876
  FVG Bias: BULLISH_BIAS      ← FVG H1
  Active FVGs: 3
  RSI: 28.45                  ← RSI M15
  Volume Ratio: 1.67x

...

Total Signals: 15
  BUY: 8
  SELL: 7
```

---

## 📖 Advanced: Custom Strategy

Nếu muốn strategy phức tạp hơn, edit file:

**`strategies/multi_timeframe_strategy.py`**

### **Method: `_generate_signal()`**

```python
def _generate_signal(self, fvg_analysis: dict, indicators: dict) -> str:
    """
    CUSTOM LOGIC Ở ĐÂY!
    """
    primary_fvg = fvg_analysis.get('primary')
    secondary_fvg = fvg_analysis.get('secondary')  # Nếu có

    # Example: Require both H1 and H4 FVG bullish
    if (primary_fvg and secondary_fvg and
        primary_fvg['bias'] == 'BULLISH_BIAS' and
        secondary_fvg['bias'] == 'BULLISH_BIAS' and
        indicators['rsi'] < 30):
        return 'BUY'

    # Example: MACD crossover
    if (primary_fvg['bias'] == 'BULLISH_BIAS' and
        indicators['macd'] > indicators['macd_signal']):
        return 'BUY'

    return 'NEUTRAL'
```

---

## 🚀 Quick Reference

### **Chỉnh timeframe:**
1. Mở `config.py`
2. Tìm `MULTI_TIMEFRAME_STRATEGY_CONFIG`
3. Đổi giá trị timeframe
4. Save
5. Chạy: `python strategies/multi_timeframe_strategy.py`

### **Supported timeframes:**
`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`, `MN1`

### **Files quan trọng:**
- `config.py` - **CONFIG Ở ĐÂY!** ⭐
- `strategies/multi_timeframe_strategy.py` - Strategy template
- `core/fvg/multi_timeframe_manager.py` - FVG multi-timeframe engine

---

## 📝 Tóm tắt

**Câu hỏi ban đầu:**
> "Sau này test thì tôi vô file nào để chỉnh timeframe?"

**Trả lời:**
✅ **File:** `config.py`
✅ **Section:** `MULTI_TIMEFRAME_STRATEGY_CONFIG`
✅ **Chỉnh:** Đổi giá trị timeframe cho từng indicator/FVG
✅ **Chạy:** `python strategies/multi_timeframe_strategy.py`

**Tất cả config tập trung 1 chỗ, dễ dàng thay đổi!**

---

**CREATED:** 2025-10-24
**STATUS:** ✅ READY TO USE
