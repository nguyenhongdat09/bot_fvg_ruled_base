# PHAN TICH: FVG MODULE - NO LOOK-AHEAD BIAS

## 1. VAN DE USER HOI

> "CSV export ra co FVG da touched (index 22), vay khi strategy test tai index 20,
> no co bi leak thong tin tuong lai khong?"

**Tra loi: KHONG BI LOOK-AHEAD BIAS**

---

## 2. TAI SAO KHONG BI LOOK-AHEAD BIAS?

### 2.1. CSV Export vs Strategy API

#### CSV Export (export_history_to_dataframe)
```python
# File: core/fvg/fvg_manager.py:305-319

def export_history_to_dataframe(self) -> pd.DataFrame:
    """
    Export ALL FVG history to DataFrame
    -> FINAL STATE (sau khi backtest xong)
    -> CHI DE REVIEW/ANALYSIS
    """
    records = [fvg.to_dict() for fvg in self.all_fvgs_history]
    return pd.DataFrame(records)
```

**Du lieu CSV ban thay:**
```csv
fvg_id,created_index,touched_index,is_touched
BEARISH_20250428_050000_20,20,22,TRUE
```
- Index 20: FVG tao ra
- Index 22: FVG bi touched
- `is_touched=TRUE` la **final state** (sau khi backtest xong)

**Muc dich:** De xem lai lich su, KHONG dung cho strategy testing!

---

#### Strategy API (get_market_structure)
```python
# File: core/fvg/fvg_manager.py:181-260

def get_market_structure(self, current_price: float) -> Dict:
    """
    Return ACTIVE FVGs (is_touched=False) tai thoi diem hien tai
    -> REAL-TIME STATE
    -> DUNG CHO STRATEGY TESTING
    """
    bullish_fvgs_below = [
        fvg for fvg in self.active_bullish_fvgs  # CHI LAY ACTIVE
        if fvg.is_valid_target(current_price)
    ]

    return {
        'bias': bias,
        'bullish_fvgs_below': bullish_fvgs_below,
        'bearish_fvgs_above': bearish_fvgs_above,
        ...
    }
```

**Strategy chi thay:**
- **Index 20**: FVG vua tao, `is_touched=False` -> **ACTIVE** ✓
- **Index 21**: FVG chua touched -> **ACTIVE** ✓
- **Index 22**: FVG bi touched -> **REMOVED** khoi active_list ✓

---

### 2.2. Flow xu ly tuan tu (Sequential Processing)

```python
# File: test_fvg_real_data.py:120-125

for i in range(20, len(data)):
    # CRITICAL: Chi truyen data tu start -> i (KHONG CO TUONG LAI!)
    manager.update(data.iloc[:i+1], i, atr.iloc[i])

    # Get market structure TAI INDEX i
    structure = manager.get_market_structure(data.iloc[i]['close'])
    # structure chi chua ACTIVE FVGs tai thoi diem i
```

**Vi du cu the:**

**Index 20** (FVG vua tao):
```python
manager.update(data.iloc[:21], 20, atr.iloc[20])
# -> Tao FVG moi: is_touched=False
# -> active_bullish_fvgs = [FVG_20]

structure = manager.get_market_structure(price_20)
# -> Tra ve: {'bias': 'BULLISH_BIAS', 'bullish_fvgs_below': [FVG_20]}
# Strategy thay: FVG ACTIVE
```

**Index 21**:
```python
manager.update(data.iloc[:22], 21, atr.iloc[21])
# -> Check FVG_20: chua touched
# -> active_bullish_fvgs = [FVG_20]  (van con)

structure = manager.get_market_structure(price_21)
# Strategy thay: FVG van ACTIVE
```

**Index 22** (FVG bi touched):
```python
manager.update(data.iloc[:23], 22, atr.iloc[22])
# -> Check FVG_20: BI TOUCHED!
# -> _update_fvg_states(): set is_touched=True
# -> _remove_touched_fvgs(): REMOVE khoi active_bullish_fvgs
# -> active_bullish_fvgs = []

structure = manager.get_market_structure(price_22)
# Strategy thay: KHONG CON FVG (vi da bi touched)
```

---

### 2.3. Critical Code: Skip checking creation candle

```python
# File: core/fvg/fvg_manager.py:124-126, 136-137

def _update_fvg_states(self, current_candle, current_index, current_timestamp):
    for fvg in self.active_bullish_fvgs:
        # Skip FVG vua moi tao tai current_index
        if fvg.created_index == current_index:
            continue  # CRITICAL: Khong check touched voi nen tao no

        was_touched = fvg.check_touched(...)
```

**Tai sao quan trong?**
- Tranh bug: FVG bi danh dau touched ngay khi tao
- Dam bao: FVG active it nhat 1 candle sau khi tao

---

## 3. KET LUANAN: CO DU DIEU KIEN KET HOP VOI INDICATOR KHAC KHONG?

### ✅ FVG Module da dam bao NO LOOK-AHEAD BIAS vi:

1. **Sequential Processing**: Xu ly tuan tu, chi dung data qua khu
2. **Real-time State**: Strategy chi access active FVGs (current state)
3. **Immediate Update**: FVG touched -> remove ngay lap tuc
4. **CSV Export != Strategy Input**: CSV la final state de review, khong feed vao strategy

---

### ⚠️ DIEU KIEN khi ket hop voi Indicator khac:

#### 1. Indicator phai xu ly tuan tu (Sequential)

**✅ DUNG:**
```python
# File: indicators/volume.py (se viet)

for i in range(lookback, len(data)):
    # CHI DUNG data tu 0 -> i
    volume_sma = data.iloc[:i+1]['volume'].rolling(20).mean().iloc[-1]

    # Hoac dung indicator da tinh san (pandas_ta)
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    current_volume_sma = df.iloc[i]['volume_sma']
```

**❌ SAI (Look-ahead bias):**
```python
# KHONG BAO GIO DUNG .shift(-1) hoac future data!
df['next_close'] = df['close'].shift(-1)  # ❌ LAY DU LIEU TUONG LAI!

# KHONG dung .iloc[i+1:] khi test tai index i
future_data = data.iloc[i+1:]  # ❌ DU LIEU TUONG LAI!
```

---

#### 2. Indicator library phai an toan

**✅ AN TOAN:**
- `pandas.rolling()`: An toan, chi dung data qua khu
- `pandas_ta`: An toan, indicators da optimize
- `ta-lib`: An toan (neu dung dung)

**Vi du:**
```python
import pandas_ta as ta

# Tinh ATR (Average True Range)
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

# Tai index i, df.iloc[i]['atr'] CHI DUNG data tu i-13 den i
```

**❌ KHONG AN TOAN:**
```python
# Custom indicator MA KHONG check future data
def custom_indicator(data, i):
    # SAI: Dung data tuong lai
    future_peak = data.iloc[i:i+10]['high'].max()  # ❌
    return future_peak
```

---

#### 3. Cach ket hop FVG + Indicator dung

**Vi du: FVG + RSI + Volume**

```python
# Step 1: Prepare indicators (toan bo data)
df['rsi'] = ta.rsi(df['close'], length=14)
df['volume_sma'] = ta.sma(df['volume'], length=20)
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

# Step 2: Sequential backtesting
for i in range(100, len(df)):
    # === FVG update (chi dung data qua khu) ===
    manager.update(df.iloc[:i+1], i, df.iloc[i]['atr'])

    # === Get FVG structure tai index i ===
    structure = manager.get_market_structure(df.iloc[i]['close'])

    # === Get indicator values tai index i ===
    rsi = df.iloc[i]['rsi']
    volume = df.iloc[i]['volume']
    volume_sma = df.iloc[i]['volume_sma']

    # === Strategy logic (KHONG CO TUONG LAI!) ===
    if structure['bias'] == 'BULLISH_BIAS' and rsi < 30 and volume > volume_sma * 1.5:
        signal = 'BUY'
        # Tat ca data deu tu qua khu (index 0 -> i)
```

**Tai sao dung?**
- `df['rsi']` tinh toan toan bo truoc, nhung CHI ACCESS `df.iloc[i]['rsi']` (gia tri tai i)
- RSI tai index i CHI DUNG close prices tu `i-13` den `i`
- FVG manager CHI xu ly data `df.iloc[:i+1]`
- **KHONG CO DATA TUONG LAI nao duoc dung!**

---

#### 4. Test kiem tra Look-ahead Bias

```python
# File: tests/test_no_look_ahead.py (nen viet)

def test_no_look_ahead_bias():
    """
    Test: Ket qua backtest KHONG THAY DOI khi them data tuong lai
    """
    # Test 1: Backtest voi 1000 candles
    results_1000 = backtest(data[:1000])

    # Test 2: Backtest voi 1500 candles (nhung CHI LAY ket qua 1000 candles dau)
    results_1500 = backtest(data[:1500])[:1000]

    # Ket qua phai GIONG NHAU HOAN TOAN!
    assert results_1000 == results_1500
    # Neu khac nhau => CO LOOK-AHEAD BIAS!
```

---

## 4. KET LUAN VA KHUYEN NGHI

### ✅ FVG Module READY cho production:

1. **CSV export** la final state (de review)
2. **Strategy API** (`get_market_structure`) la real-time state (de trading)
3. **No look-ahead bias** da verify bang `demo_fvg_time_accuracy.py`
4. **Ready** de ket hop voi indicators khac

### 📋 Checklist khi phat trien Indicator module:

- [ ] Indicator xu ly tuan tu (sequential)
- [ ] KHONG dung `.shift(-1)` hoac future data
- [ ] CHI access `df.iloc[i]` khi test tai index `i`
- [ ] Use pandas/pandas_ta/ta-lib (da tested, an toan)
- [ ] Viet unit test kiem tra look-ahead bias
- [ ] Document ro rang cach su dung indicator

### 🎯 Buoc tiep theo:

1. **Phat trien Indicator module** voi cac indicator:
   - Volume (Volume SMA, Volume Spike detection)
   - Momentum (RSI, Stochastic)
   - Volatility (ATR, Bollinger Bands)
   - Trend (EMA, SMA, MACD)

2. **Strategy module** ket hop FVG + Indicators:
   - Confluence scoring (diem hop luu)
   - Signal generation (tao tin hieu)
   - Risk management (quan ly rui ro)

3. **Backtest module**:
   - Virtual engine (che do ao)
   - Real engine (che do that)
   - Martingale money management

---

## 5. TAI LIEU THAM KHAO

- `core/fvg/fvg_manager.py` - Main logic
- `demo_fvg_time_accuracy.py` - Proof no look-ahead bias
- `FVG_TIME_ACCURACY_PROOF.md` - Chi tiet ve time accuracy
- `test_fvg_real_data.py` - Test voi real MT5 data

---

**CREATED:** 2025-10-24
**AUTHOR:** Claude Code
**STATUS:** ✅ VERIFIED - FVG Module ready cho production
