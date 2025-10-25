# HƯỚNG DẪN SỬ DỤNG INDICATORS

## 📖 Tổng quan

Module indicators đã implement **Week 1-4** đầy đủ:
- ✅ Week 1: ATR + Volume Analysis
- ✅ Week 2: VWAP + OBV
- ✅ Week 3: Confluence Scorer
- ✅ Week 4: ADX Filter

**Architecture:** Dễ dàng thêm/bớt indicators sau này!

---

## 🎯 INDICATORS ĐÃ IMPLEMENT

| Indicator | Purpose | Score Weight | Status |
|-----------|---------|--------------|--------|
| **FVG** | Primary signal (trend direction) | 50% | ✅ (core/fvg) |
| **VWAP** | Volume confirmation | 20% | ✅ |
| **OBV** | Volume trend | 15% | ✅ |
| **Volume Spike** | Confirmation | 15% | ✅ |
| **ATR** | Risk management | N/A | ✅ |
| **ADX** | Trend filter (optional) | Filter | ✅ |

**Total Score:** 100%

---

## 🚀 QUICK START

### **1. Basic Usage**

```python
from indicators import (
    ATRIndicator, VWAPIndicator, OBVIndicator,
    VolumeAnalyzer, ADXIndicator, ConfluenceScorer
)
import pandas as pd

# Load data
data = pd.read_csv('data/EURUSD_M15_180days.csv', index_col=0, parse_dates=True)

# === Individual Indicators ===

# ATR
atr = ATRIndicator(period=14)
atr_values = atr.calculate(data)
print(f"Current ATR: {atr_values.iloc[-1]:.5f}")

# VWAP
vwap = VWAPIndicator()
vwap_values = vwap.calculate(data)
print(f"Current VWAP: {vwap_values.iloc[-1]:.5f}")

# OBV
obv = OBVIndicator()
obv_values = obv.calculate(data)
print(f"Current OBV: {obv_values.iloc[-1]:.0f}")

# Volume
vol = VolumeAnalyzer(period=20)
vol_analysis = vol.calculate(data)
print(f"Volume spike: {vol_analysis.iloc[-1]['is_spike']}")

# ADX
adx = ADXIndicator(period=14)
adx_data = adx.calculate(data)
print(f"Current ADX: {adx_data.iloc[-1]['adx']:.2f}")
```

---

### **2. Confluence Scoring**

```python
from core.fvg.fvg_manager import FVGManager
from indicators import ConfluenceScorer, ATRIndicator

# Initialize
fvg_manager = FVGManager()
scorer = ConfluenceScorer()
atr = ATRIndicator()

# Calculate ATR
atr_values = atr.calculate(data)

# Backtest
for i in range(100, len(data)):
    # Update FVG
    fvg_manager.update(data.iloc[:i+1], i, atr_values.iloc[i])
    fvg_structure = fvg_manager.get_market_structure(data.iloc[i]['close'])

    # Calculate score
    result = scorer.calculate_score(
        data=data.iloc[:i+1],
        index=i,
        fvg_structure=fvg_structure,
        atr_value=atr_values.iloc[i]
    )

    # Check signal
    if result['should_trade']:
        print(f"{result['signal']} at {data.index[i]}")
        print(f"  Score: {result['total_score']:.0f}% ({result['confidence']})")
        print(f"  SL: {result['sl_tp']['sl']:.5f}")
        print(f"  TP: {result['sl_tp']['tp']:.5f}")
```

---

## 📊 CONFLUENCE SCORING SYSTEM

### **Default Weights:**

```python
{
    'fvg': 50,      # Primary signal from FVG
    'vwap': 20,     # Volume confirmation
    'obv': 15,      # Volume trend
    'volume': 15,   # Spike confirmation
}
# Total: 100%
```

### **Score Breakdown:**

#### **1. FVG (50 points)**
```python
if fvg_bias == 'BULLISH_BIAS':
    score += 50
    signal = 'BUY'
elif fvg_bias == 'BEARISH_BIAS':
    score += 50
    signal = 'SELL'
else:
    score = 0  # No trade
```

#### **2. VWAP (20 points)**
```python
# BUY: price > VWAP (institutional bullish)
if signal == 'BUY' and price > vwap:
    score += 20

# SELL: price < VWAP (institutional bearish)
if signal == 'SELL' and price < vwap:
    score += 20
```

#### **3. OBV (15 points)**
```python
# BUY: OBV > SMA(20) (accumulation)
if signal == 'BUY' and obv > obv_sma:
    score += 15

# SELL: OBV < SMA(20) (distribution)
if signal == 'SELL' and obv < obv_sma:
    score += 15
```

#### **4. Volume Spike (15 points)**
```python
volume_ratio = current_volume / avg_volume

if volume_ratio > 2.0:
    score += 15  # Very strong spike
elif volume_ratio > 1.5:
    score += 10  # Strong spike
elif volume_ratio > 1.2:
    score += 5   # Moderate spike
```

#### **5. ADX Filter (Optional)**
```python
if adx < 25:
    skip_trade()  # Ranging market
```

---

### **Signal Confidence:**

| Score | Confidence | Action |
|-------|------------|--------|
| **≥ 70%** | HIGH | Trade with 2% risk |
| **60-70%** | MEDIUM | Trade with 1% risk |
| **< 60%** | LOW | Skip |

---

## 🔧 CUSTOMIZATION

### **1. Change Weights**

```python
# More weight on FVG
custom_weights = {
    'fvg': 60,      # Increase FVG importance
    'vwap': 20,
    'obv': 10,      # Decrease OBV
    'volume': 10
}

scorer = ConfluenceScorer(weights=custom_weights)
```

---

### **2. Enable/Disable ADX Filter**

```python
# Enable ADX filter
scorer = ConfluenceScorer(adx_enabled=True, adx_threshold=25)

# Disable ADX filter
scorer = ConfluenceScorer(adx_enabled=False)

# Or toggle after creation
scorer.enable_adx_filter(threshold=25)
scorer.disable_adx_filter()
```

---

### **3. Add New Indicator**

**Example: Add RSI indicator**

```python
# Step 1: Create new indicator class
# File: indicators/momentum.py

from .base import BaseIndicator
import pandas as pd

class RSIIndicator(BaseIndicator):
    """RSI (Relative Strength Index)"""

    def __init__(self, period=14):
        super().__init__(name='RSI', period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI"""
        close = data['close']
        delta = close.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

# Step 2: Add to __init__.py
# File: indicators/__init__.py
from .momentum import RSIIndicator

# Step 3: Use in confluence scorer
# Modify confluence.py _score_rsi() method

def _score_rsi(self, data, index, signal):
    """Score RSI confirmation"""
    if index < 14:
        return 0

    rsi_values = self.rsi.calculate(data)
    current_rsi = rsi_values.iloc[index]

    if signal == 'BUY' and current_rsi < 30:
        return 10  # Oversold
    elif signal == 'SELL' and current_rsi > 70:
        return 10  # Overbought

    return 0

# Step 4: Update weights
custom_weights = {
    'fvg': 50,
    'vwap': 15,
    'obv': 10,
    'volume': 15,
    'rsi': 10  # New component
}
```

---

## 📋 INDICATOR REFERENCE

### **ATRIndicator**

```python
class ATRIndicator(BaseIndicator):
    """
    ATR (Average True Range) - Volatility measurement

    Methods:
        calculate(data) -> pd.Series
        get_position_size(account, risk_pct, atr, multiplier) -> float
        get_sl_tp_levels(entry, atr, direction, sl_mult, tp_mult) -> dict
    """

# Usage:
atr = ATRIndicator(period=14)
atr_values = atr.calculate(data)

# Position sizing
lot_size = atr.get_position_size(10000, 0.01, atr_values.iloc[-1])

# SL/TP levels
levels = atr.get_sl_tp_levels(1.10000, atr_values.iloc[-1], 'BUY')
# Returns: {'sl': 1.09925, 'tp': 1.10150, 'rr_ratio': 2.0, ...}
```

---

### **VWAPIndicator**

```python
class VWAPIndicator(BaseIndicator):
    """
    VWAP (Volume Weighted Average Price) - Institutional benchmark

    Methods:
        calculate(data) -> pd.Series
        get_bands(data, std_multiplier) -> pd.DataFrame
    """

# Usage:
vwap = VWAPIndicator()
vwap_values = vwap.calculate(data)

# With bands (similar to Bollinger)
bands = vwap.get_bands(data, std_multiplier=1.0)
# Returns: DataFrame with [vwap, upper_band, lower_band]
```

---

### **OBVIndicator**

```python
class OBVIndicator(BaseIndicator):
    """
    OBV (On-Balance Volume) - Cumulative volume trend

    Methods:
        calculate(data) -> pd.Series
        get_obv_sma(data, period) -> pd.DataFrame
        detect_divergence(data, lookback) -> pd.Series
    """

# Usage:
obv = OBVIndicator()
obv_values = obv.calculate(data)

# With SMA for trend
obv_data = obv.get_obv_sma(data, period=20)
# Returns: DataFrame with [obv, obv_sma, obv_above_sma]

# Divergence detection
divergence = obv.detect_divergence(data, lookback=14)
# Returns: Series (1=bullish, -1=bearish, 0=none)
```

---

### **VolumeAnalyzer**

```python
class VolumeAnalyzer(BaseIndicator):
    """
    Volume Analysis - Spike detection

    Methods:
        calculate(data) -> pd.DataFrame
        get_volume_score(current_vol, avg_vol) -> int
    """

# Usage:
vol = VolumeAnalyzer(period=20, spike_threshold=1.5)
vol_analysis = vol.calculate(data)
# Returns: DataFrame with [avg_volume, volume_ratio, is_spike, spike_strength]

# Get score
score = vol.get_volume_score(current_vol, avg_vol)
# Returns: 0-15 points
```

---

### **ADXIndicator**

```python
class ADXIndicator(BaseIndicator):
    """
    ADX (Average Directional Index) - Trend strength

    Methods:
        calculate(data) -> pd.DataFrame
        is_trending(data, threshold) -> pd.Series
        get_trend_direction(data) -> pd.Series
    """

# Usage:
adx = ADXIndicator(period=14)
adx_data = adx.calculate(data)
# Returns: DataFrame with [adx, plus_di, minus_di, trend_strength]

# Check if trending
is_trending = adx.is_trending(data, threshold=25)
# Returns: Boolean Series

# Get direction
direction = adx.get_trend_direction(data)
# Returns: Series ('BULLISH', 'BEARISH', 'NEUTRAL')
```

---

### **ConfluenceScorer**

```python
class ConfluenceScorer:
    """
    Confluence Scoring System

    Methods:
        calculate_score(data, index, fvg_structure, atr_value) -> dict
        update_weights(new_weights) -> None
        get_weights() -> dict
        enable_adx_filter(threshold) -> None
        disable_adx_filter() -> None
    """

# Usage:
scorer = ConfluenceScorer(
    weights={'fvg': 50, 'vwap': 20, 'obv': 15, 'volume': 15},
    adx_enabled=True,
    adx_threshold=25
)

result = scorer.calculate_score(data, index, fvg_structure, atr_value)

# Returns:
{
    'total_score': 85,
    'signal': 'BUY',
    'confidence': 'HIGH',
    'components': {'fvg': 50, 'vwap': 20, 'obv': 15, 'volume': 0},
    'should_trade': True,
    'reason': 'High confluence (85%)',
    'sl_tp': {'sl': 1.09925, 'tp': 1.10150, 'rr_ratio': 2.0}
}
```

---

## 📈 STRATEGY EXAMPLES

### **Example 1: High Confidence Only**

```python
scorer = ConfluenceScorer()

for i in range(100, len(data)):
    result = scorer.calculate_score(...)

    if result['confidence'] == 'HIGH':  # >= 70%
        execute_trade(result['signal'], risk=0.02)  # 2% risk
```

---

### **Example 2: Varying Risk by Confidence**

```python
for i in range(100, len(data)):
    result = scorer.calculate_score(...)

    if result['confidence'] == 'HIGH':
        risk = 0.02  # 2% risk
    elif result['confidence'] == 'MEDIUM':
        risk = 0.01  # 1% risk
    else:
        continue  # Skip LOW confidence

    lot_size = calculate_position_size(account, risk, result['sl_tp'])
    execute_trade(result['signal'], lot_size)
```

---

### **Example 3: Trending Markets Only**

```python
scorer = ConfluenceScorer(adx_enabled=True, adx_threshold=25)

# Scorer will automatically skip signals when ADX < 25
for i in range(100, len(data)):
    result = scorer.calculate_score(...)

    if result['should_trade']:
        execute_trade(result['signal'])
    # Signals in ranging markets (ADX < 25) will have should_trade=False
```

---

## 🧪 TESTING

### **Run Example:**

```bash
python examples/indicators_example.py
```

**Output:**
```
================================
EXAMPLE 1: BASIC INDICATORS
================================

--- ATR (Average True Range) ---
Current ATR: 0.00052
Position Sizing (Account: $10,000, Risk: 1%):
  Lot size: 96.15

--- VWAP ---
Current VWAP: 1.09876
Current Price: 1.09920
Price vs VWAP: ABOVE (Bullish)

--- OBV ---
Current OBV: 1234567
OBV Trend: RISING (Accumulation)

...

================================
EXAMPLE 2: CONFLUENCE SCORING
================================

BUY Signal at 2024-10-15 10:30:00
  Score: 85.0/100 (HIGH)
  Components:
    fvg: 50.0
    vwap: 20.0
    obv: 15.0
    volume: 0.0
  SL: 1.09780
  TP: 1.10156

...
```

---

## 📁 FILE STRUCTURE

```
indicators/
├── __init__.py           # Module exports
├── base.py               # BaseIndicator, IndicatorRegistry
├── volatility.py         # ATR
├── volume.py             # VWAP, OBV, VolumeAnalyzer
├── trend.py              # ADX
└── confluence.py         # ConfluenceScorer

examples/
└── indicators_example.py # Usage examples
```

---

## 🔄 EXTENSIBILITY

### **Architecture Allows:**

1. **Add new indicators** - Inherit from `BaseIndicator`
2. **Custom scoring logic** - Modify `ConfluenceScorer` methods
3. **Dynamic weights** - Configure via `update_weights()`
4. **Registry system** - Auto-discover indicators via `IndicatorRegistry`

### **Future Indicators (Easy to add):**

```python
# indicators/momentum.py
class RSIIndicator(BaseIndicator):
    """RSI - Momentum oscillator"""
    pass

class MFIIndicator(BaseIndicator):
    """MFI - Money Flow Index"""
    pass

# indicators/trend.py
class SupertrendIndicator(BaseIndicator):
    """Supertrend - Trend follower"""
    pass

class IchimokuIndicator(BaseIndicator):
    """Ichimoku - Complete system"""
    pass

# indicators/volume.py
class CMFIndicator(BaseIndicator):
    """CMF - Chaikin Money Flow"""
    pass

class VolumeProfileIndicator(BaseIndicator):
    """Volume Profile - POC, HVN, LVN"""
    pass
```

**Chỉ cần inherit `BaseIndicator` là tự động integrate vào system!**

---

## ✅ SUMMARY

### **Đã implement:**
- ✅ Week 1-4 FULL (ATR, VWAP, OBV, Volume, ADX, Confluence)
- ✅ Extensible architecture (dễ thêm indicators mới)
- ✅ Config-driven weights
- ✅ Complete scoring system (0-100%)
- ✅ Examples & documentation

### **Chưa implement (có thể thêm sau):**
- ⚠️ RSI, MACD (nếu muốn làm confirmation)
- ⚠️ Supertrend, Ichimoku (nếu muốn advanced)
- ⚠️ Volume Profile (nếu muốn POC/HVN/LVN)

### **Khuyến nghị:**
**KHÔNG CẦN THÊM GÌ NỮA!** Core set (FVG + VWAP + OBV + Volume + ATR + ADX) đã ĐỦ cho strategy!

---

**CREATED:** 2025-10-24
**STATUS:** ✅ FULLY IMPLEMENTED
