# MULTI-TIMEFRAME ANALYSIS - STRUCTURE HIỆN TẠI VÀ GIẢI PHÁP

## 1. CÂU HỎI CỦA USER

> "Sau này tôi muốn phân tích FVG ở khung H1, nhưng các indicator khác lại chọn
> khung M15, hoặc mỗi indicator một khung thời gian khác nhau.
> Structure hiện tại có đáp ứng được không?"

**VÍ DỤ:**
- FVG: Phân tích trên **H1** (1 hour)
- RSI: Phân tích trên **M15** (15 minutes)
- Volume: Phân tích trên **M5** (5 minutes)
- MACD: Phân tích trên **H4** (4 hours)

---

## 2. PHÂN TÍCH STRUCTURE HIỆN TẠI

### 2.1. FVGManager hiện tại

```python
# core/fvg/fvg_manager.py

class FVGManager:
    def __init__(self, lookback_days=90, min_gap_atr_ratio=0.3):
        # KHÔNG CÓ timeframe parameter
        pass

    def update(self, data: pd.DataFrame, current_index: int, atr: float):
        """
        Input: DataFrame với index là timestamp
        - data.iloc[:current_index+1] là data từ start đến current
        - Giả định: data là 1 timeframe duy nhất
        """
        pass
```

**Giới hạn:**
- ❌ Chỉ xử lý 1 timeframe duy nhất
- ❌ Không có cơ chế resample data
- ❌ Không có cơ chế sync giữa các timeframe

---

### 2.2. Vấn đề khi Multi-Timeframe

**Scenario: FVG H1 + RSI M15**

```
H1:  |-----Candle 1-----|-----Candle 2-----|-----Candle 3-----|
     10:00              11:00              12:00              13:00

M15: |C1|C2|C3|C4|C5|C6|C7|C8|C9|C10|C11|C12|
     10:00  10:30  11:00  11:30  12:00  12:30  13:00
```

**Vấn đề:**
1. **Index mismatch**: M15 có 4 candles trong 1 H1 candle
2. **Sync issue**: Khi iterate M15, H1 FVG state nào được dùng?
3. **Look-ahead risk**: Nếu không cẩn thận, có thể leak future data!

**Ví dụ cụ thể:**
```python
# M15 index 100 = 10:00
# Cần FVG H1 state tại 10:00
# Nhưng H1 index = 100/4 = 25
# -> Làm sao map chính xác?
```

---

## 3. ĐÁNH GIÁ: STRUCTURE HIỆN TẠI CÓ ĐÁP ỨNG KHÔNG?

### ❌ **Trả lời ngắn gọn: CHƯA ĐÁP ỨNG**

**Lý do:**
1. FVGManager không nhận timeframe parameter
2. Không có cơ chế resample/align data giữa các timeframe
3. Không có API để query state tại arbitrary timestamp

### ✅ **Nhưng CÓ THỂ EXTEND DỄ DÀNG!**

**Lý do:**
1. Sequential processing vẫn đảm bảo no look-ahead bias
2. Timestamp-based index → dễ align
3. Có thể wrap vào MultiTimeframeManager

---

## 4. GIẢI PHÁP ĐỀ XUẤT

### 🎯 **Option 1: Resample & Align (ĐƠN GIẢN - KHUYẾN NGHỊ)**

**Ý tưởng:**
- Chọn timeframe nhỏ nhất làm base (M15)
- Resample lên các timeframe lớn hơn (H1, H4...)
- Forward-fill để align về base timeframe
- Iterate qua base timeframe, access các timeframe khác qua aligned data

**Ưu điểm:**
- ✅ Đơn giản, dễ implement
- ✅ Không cần thay đổi FVGManager
- ✅ Linh hoạt, dễ debug
- ✅ Tường minh, dễ hiểu

**Nhược điểm:**
- ⚠️ Cần manual resample và align
- ⚠️ Code hơi dài (nhưng clear)

**Code example:**
```python
# Step 1: Load M15 data (base timeframe)
m15_data = load_mt5_data('EURUSD', 'M15', 30)

# Step 2: Resample to H1
h1_data = m15_data.resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Step 3: Calculate indicators on respective timeframes
# FVG on H1
fvg_h1 = FVGManager()
h1_data['atr'] = ta.atr(h1_data['high'], h1_data['low'], h1_data['close'], 14)

# Process H1 sequentially
h1_states = []
for i in range(20, len(h1_data)):
    fvg_h1.update(h1_data.iloc[:i+1], i, h1_data.iloc[i]['atr'])
    structure = fvg_h1.get_market_structure(h1_data.iloc[i]['close'])
    h1_states.append({
        'timestamp': h1_data.index[i],
        'bias': structure['bias'],
        'fvg_count': structure['total_active_fvgs']
    })

# Convert to DataFrame for easy lookup
h1_states_df = pd.DataFrame(h1_states).set_index('timestamp')

# RSI on M15
m15_data['rsi'] = ta.rsi(m15_data['close'], length=14)

# Step 4: Align H1 states to M15 (forward fill)
m15_data['h1_bias'] = None
m15_data['h1_fvg_count'] = None

for i in range(len(m15_data)):
    m15_time = m15_data.index[i]

    # Find latest H1 state BEFORE or AT m15_time (forward fill)
    h1_time = h1_states_df.index[h1_states_df.index <= m15_time]

    if len(h1_time) > 0:
        latest_h1_time = h1_time[-1]
        m15_data.loc[m15_time, 'h1_bias'] = h1_states_df.loc[latest_h1_time, 'bias']
        m15_data.loc[m15_time, 'h1_fvg_count'] = h1_states_df.loc[latest_h1_time, 'fvg_count']

# Step 5: Strategy on M15 with H1 FVG
for i in range(100, len(m15_data)):
    # H1 FVG (aligned to M15)
    h1_bias = m15_data.iloc[i]['h1_bias']

    # M15 RSI
    m15_rsi = m15_data.iloc[i]['rsi']

    # Strategy
    if h1_bias == 'BULLISH_BIAS' and m15_rsi < 30:
        signal = 'BUY'
```

**Timeline minh họa:**
```
H1 States:
10:00 -> BULLISH_BIAS, 3 FVGs
11:00 -> BULLISH_BIAS, 2 FVGs
12:00 -> NO_FVG, 0 FVGs

M15 Aligned (forward fill):
10:00 -> BULLISH_BIAS, 3 FVGs  (from H1 10:00)
10:15 -> BULLISH_BIAS, 3 FVGs  (forward fill)
10:30 -> BULLISH_BIAS, 3 FVGs  (forward fill)
10:45 -> BULLISH_BIAS, 3 FVGs  (forward fill)
11:00 -> BULLISH_BIAS, 2 FVGs  (from H1 11:00)
11:15 -> BULLISH_BIAS, 2 FVGs  (forward fill)
...
```

---

### 🎯 **Option 2: MultiTimeframeManager Class (CHUYÊN NGHIỆP)**

**Ý tưởng:**
- Tạo class `MultiTimeframeManager` quản lý nhiều timeframes
- Mỗi timeframe có 1 FVGManager riêng
- Class tự động resample, align, và sync

**Ưu điểm:**
- ✅ Clean API, dễ sử dụng
- ✅ Encapsulate logic phức tạp
- ✅ Reusable cho nhiều strategies
- ✅ Professional, maintainable

**Nhược điểm:**
- ⚠️ Phức tạp hơn khi implement
- ⚠️ Cần test kỹ

**Code example:**
```python
# core/fvg/multi_timeframe_manager.py

class MultiTimeframeManager:
    """
    Manage FVGs across multiple timeframes

    Usage:
        mtf = MultiTimeframeManager(base_data, base_timeframe='M15')
        mtf.add_fvg_timeframe('H1')
        mtf.add_fvg_timeframe('H4')

        for i in range(100, len(base_data)):
            mtf.update(i)
            h1_bias = mtf.get_fvg_bias('H1', i)
            h4_bias = mtf.get_fvg_bias('H4', i)
    """

    def __init__(self, base_data: pd.DataFrame, base_timeframe: str = 'M15'):
        self.base_data = base_data
        self.base_timeframe = base_timeframe
        self.managers = {}  # {timeframe: FVGManager}
        self.resampled_data = {}  # {timeframe: DataFrame}
        self.aligned_states = {}  # {timeframe: aligned_bias_series}

    def add_fvg_timeframe(self, timeframe: str, lookback_days=90):
        """Add FVG analysis for a timeframe"""
        # Resample base_data to target timeframe
        resampled = self._resample_data(self.base_data, timeframe)
        self.resampled_data[timeframe] = resampled

        # Create FVGManager for this timeframe
        manager = FVGManager(lookback_days=lookback_days)
        self.managers[timeframe] = manager

        # Initialize aligned states
        self.aligned_states[timeframe] = pd.Series(index=self.base_data.index, dtype=object)

    def update(self, base_index: int):
        """Update all timeframes up to base_index"""
        base_time = self.base_data.index[base_index]

        # Update each timeframe
        for tf, manager in self.managers.items():
            resampled = self.resampled_data[tf]

            # Find corresponding index in resampled timeframe
            tf_idx = resampled.index.get_indexer([base_time], method='ffill')[0]

            if tf_idx >= 0:
                # Update manager
                manager.update(resampled.iloc[:tf_idx+1], tf_idx,
                              resampled.iloc[tf_idx]['atr'])

                # Get structure
                structure = manager.get_market_structure(resampled.iloc[tf_idx]['close'])

                # Store aligned state
                self.aligned_states[tf].loc[base_time] = structure['bias']

    def get_fvg_bias(self, timeframe: str, base_index: int) -> str:
        """Get FVG bias for a timeframe at base_index"""
        base_time = self.base_data.index[base_index]
        return self.aligned_states[timeframe].loc[base_time]

    def get_fvg_structure(self, timeframe: str, base_index: int) -> Dict:
        """Get full FVG structure for a timeframe"""
        # Similar to get_fvg_bias but return full structure
        pass

# Usage:
mtf = MultiTimeframeManager(m15_data, base_timeframe='M15')
mtf.add_fvg_timeframe('H1')
mtf.add_fvg_timeframe('H4')

for i in range(100, len(m15_data)):
    mtf.update(i)

    h1_bias = mtf.get_fvg_bias('H1', i)
    h4_bias = mtf.get_fvg_bias('H4', i)
    m15_rsi = m15_data.iloc[i]['rsi']

    if h1_bias == 'BULLISH_BIAS' and h4_bias == 'BULLISH_BIAS' and m15_rsi < 30:
        signal = 'BUY'
```

---

### 🎯 **Option 3: Indicator MultiTimeframe Wrapper**

**Ý tưởng:**
- Không chỉ FVG, mà TẤT CẢ indicators đều hỗ trợ multi-timeframe
- Tạo wrapper class cho indicators

**Code example:**
```python
# indicators/multi_timeframe_indicator.py

class MultiTimeframeIndicator:
    """
    Wrapper for any indicator to support multiple timeframes

    Usage:
        mtf_rsi = MultiTimeframeIndicator(base_data, 'M15')
        mtf_rsi.add_timeframe('M15', lambda df: ta.rsi(df['close'], 14))
        mtf_rsi.add_timeframe('H1', lambda df: ta.rsi(df['close'], 14))

        for i in range(100, len(base_data)):
            m15_rsi = mtf_rsi.get_value('M15', i)
            h1_rsi = mtf_rsi.get_value('H1', i)
    """
    pass
```

---

## 5. SO SÁNH CÁC OPTIONS

| Tiêu chí | Option 1: Resample & Align | Option 2: MultiTFManager | Option 3: Indicator Wrapper |
|----------|---------------------------|--------------------------|----------------------------|
| **Độ phức tạp** | ⭐⭐ Đơn giản | ⭐⭐⭐⭐ Phức tạp | ⭐⭐⭐⭐⭐ Rất phức tạp |
| **Dễ implement** | ✅ Dễ | ⚠️ Trung bình | ❌ Khó |
| **Dễ debug** | ✅ Rất dễ | ⚠️ Trung bình | ❌ Khó |
| **Performance** | ⚠️ Trung bình | ✅ Tốt | ✅ Tốt |
| **Flexibility** | ⚠️ Trung bình | ✅ Tốt | ✅ Rất tốt |
| **Code reuse** | ❌ Thấp | ✅ Cao | ✅ Rất cao |
| **Khuyến nghị** | **✅ PHASE 1** | **✅ PHASE 2** | ⚠️ FUTURE |

---

## 6. KHUYẾN NGHỊ IMPLEMENTATION

### 📋 **Lộ trình phát triển:**

#### **Phase 1: Quick Start (Option 1)**
- Implement resample & align pattern
- Viết example: FVG H1 + RSI M15
- Test thoroughly
- Document pattern
- **Timeline:** 1-2 ngày

#### **Phase 2: Professional (Option 2)**
- Implement MultiTimeframeManager
- Migrate examples sang MultiTFManager
- Test với nhiều timeframes (M5, M15, H1, H4, D1)
- **Timeline:** 3-5 ngày

#### **Phase 3: Advanced (Option 3)**
- Implement indicator wrapper
- Support dynamic timeframe switching
- **Timeline:** 1 tuần

---

## 7. VERIFY NO LOOK-AHEAD BIAS TRONG MULTI-TIMEFRAME

### ⚠️ **Điểm quan trọng:**

Multi-timeframe analysis **RẤT DỄ** bị look-ahead bias nếu không cẩn thận!

**❌ SAI - Look-ahead bias:**
```python
# SAI: Resample toàn bộ data trước, sau đó iterate
h1_data = m15_data.resample('1H').agg({...})  # Resample ALL data

for i in range(100, len(m15_data)):
    m15_time = m15_data.index[i]
    h1_idx = h1_data.index.get_indexer([m15_time], method='nearest')[0]

    h1_close = h1_data.iloc[h1_idx]['close']  # ❌ Có thể lấy future H1 candle!
```

**✅ ĐÚNG - No look-ahead bias:**
```python
# ĐÚNG: Sequential processing cho từng timeframe
h1_data = m15_data.resample('1H').agg({...})

# Process H1 sequentially
for i in range(20, len(h1_data)):
    fvg_h1.update(h1_data.iloc[:i+1], i, ...)  # CHI DUNG data 0->i
    h1_states[h1_data.index[i]] = fvg_h1.get_market_structure(...)

# Align to M15 with FORWARD FILL (không bao giờ dùng future data)
for i in range(100, len(m15_data)):
    m15_time = m15_data.index[i]

    # CHI LAY H1 state TRUOC HOAC TAI m15_time
    valid_h1_times = [t for t in h1_states.keys() if t <= m15_time]
    if valid_h1_times:
        latest_h1_time = max(valid_h1_times)
        h1_bias = h1_states[latest_h1_time]['bias']
```

**Test verify:**
```python
def test_no_look_ahead_multi_timeframe():
    """
    Test: Ket qua M15 backtest khong thay doi khi them H1 data tuong lai
    """
    # Test 1: H1 co 100 candles
    results_100 = backtest_multi_tf(m15_data, h1_data[:100])

    # Test 2: H1 co 150 candles, nhung chi lay ket qua tuong ung 100 candles dau
    results_150 = backtest_multi_tf(m15_data, h1_data[:150])

    # Ket qua phai giong nhau!
    assert results_100 == results_150
```

---

## 8. KẾT LUẬN

### ❌ **Structure hiện tại:**
- Không hỗ trợ multi-timeframe out-of-the-box

### ✅ **Nhưng có thể extend:**
- **Option 1** (Resample & Align): Đơn giản, implement nhanh
- **Option 2** (MultiTFManager): Chuyên nghiệp, reusable
- **Option 3** (Indicator Wrapper): Advanced, future

### 📋 **Khuyến nghị:**
1. **Ngay bây giờ:** Implement Option 1 (1-2 ngày)
2. **Sau khi test xong:** Migrate sang Option 2 (3-5 ngày)
3. **Tương lai:** Consider Option 3 nếu cần

### 🎯 **Next Steps:**
- Tạo example: FVG H1 + RSI M15 (Option 1)
- Tạo MultiTimeframeManager class (Option 2)
- Test thoroughly với real data
- Document best practices

---

**CREATED:** 2025-10-24
**STATUS:** ✅ ANALYSIS COMPLETE - READY TO IMPLEMENT
