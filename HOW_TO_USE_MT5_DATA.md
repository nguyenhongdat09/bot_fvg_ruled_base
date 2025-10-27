# Hướng Dẫn Sử Dụng MT5 Real Data

## Tổng Quan

Hướng dẫn này giúp bạn:
1. Cấu hình MT5 paths (nếu có nhiều MT5)
2. Download data thật từ MetaTrader 5
3. Test FVG modules với real data
4. Phân tích kết quả

---

## Bước 1: Cấu Hình MT5 Path

### 1.1. Mở file `config.py`

```python
# config.py
MT5_CONFIG = {
    # THAY ĐỔI ĐƯỜNG DẪN NÀY ĐỂ CHỌN MT5
    'path': None,  # Sử dụng default

    # Hoặc chỉ định đường dẫn cụ thể:
    # 'path': r'C:\Program Files\MetaTrader 5\terminal64.exe',
}
```

### 1.2. Tìm đường dẫn MT5 của bạn

**Windows:**

Các đường dẫn phổ biến:
```
C:\Program Files\MetaTrader 5\terminal64.exe
C:\Program Files\MetaTrader 5 - IC Markets\terminal64.exe
C:\Program Files\MetaTrader 5 - XM\terminal64.exe
C:\Program Files (x86)\MetaTrader 5\terminal64.exe
```

**Cách tìm:**
1. Mở MT5
2. Help → About → See installation folder
3. Copy path đầy đủ đến `terminal64.exe`

**Linux (Wine):**
```
/home/user/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe
```

### 1.3. Cập nhật config.py

**Ví dụ 1: Sử dụng MT5 default**
```python
MT5_CONFIG = {
    'path': None,  # Auto detect
}
```

**Ví dụ 2: Chỉ định MT5 cụ thể**
```python
MT5_CONFIG = {
    'path': r'C:\Program Files\MetaTrader 5 - IC Markets\terminal64.exe',
}
```

**Ví dụ 3: Nhiều MT5 (switch dễ dàng)**
```python
# Định nghĩa các MT5 paths
MT5_PATHS = {
    'default': None,
    'ic_markets': r'C:\Program Files\MetaTrader 5 - IC Markets\terminal64.exe',
    'xm': r'C:\Program Files\MetaTrader 5 - XM\terminal64.exe',
    'exness': r'C:\Program Files\MetaTrader 5 - Exness\terminal64.exe',
}

# Chọn MT5 sử dụng
ACTIVE_MT5 = 'ic_markets'  # Thay đổi ở đây để switch

MT5_CONFIG = {
    'path': MT5_PATHS[ACTIVE_MT5],
}
```

---

## Bước 2: Download Data Từ MT5

### 2.1. Kiểm tra config hiện tại

```bash
python config.py
```

Output:
```
📁 Paths:
  MT5 Path: C:\...\terminal64.exe

📊 Data:
  Symbol: EURUSD
  Timeframe: M15
  Days: 30
```

### 2.2. Cấu hình download (optional)

Mở `config.py` và chỉnh:

```python
DATA_CONFIG = {
    'symbol': 'EURUSD',     # Cặp tiền muốn download
    'timeframe': 'M15',     # Khung thời gian
    'days': 180,             # Số ngày
}
```

**Timeframes available:**
- `M1` - 1 minute
- `M5` - 5 minutes
- `M15` - 15 minutes  ⭐ (recommended for FVG)
- `M30` - 30 minutes
- `H1` - 1 hour
- `H4` - 4 hours
- `D1` - 1 day

**Symbols phổ biến:**
- Forex: `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`
- Gold: `XAUUSD`
- Indices: `US30`, `NAS100`, `SPX500`

### 2.3. Chạy download script

```bash
python data/download_mt5_data.py
```

**Process:**
1. ✅ Connect to MT5
2. ✅ Check symbol availability
3. ✅ Download historical data
4. ✅ Convert to DataFrame
5. ✅ Save to CSV

**Output:**
```
✓ Downloaded 2880 candles
✓ Saved to: data/EURUSD_M15_30days.csv
   File size: 145.2 KB
```

### 2.4. Troubleshooting

**Lỗi: MT5 initialize() failed**
```
❌ Failed to initialize MT5
```
**Fix:**
1. Mở MT5 manually
2. Login vào account
3. Chạy lại script

**Lỗi: Symbol not found**
```
❌ Symbol EURUSD not found
```
**Fix:**
1. Check tên symbol (có thể là `EURUSDm`, `EURUSD.c`, etc.)
2. Xem available symbols:
   ```python
   import MetaTrader5 as mt5
   mt5.initialize()
   symbols = mt5.symbols_get()
   for s in symbols:
       print(s.name)
   ```

**Lỗi: No data received**
```
❌ No data received
```
**Fix:**
1. Check date range (không quá xa quá khứ)
2. Check account có quyền download data không
3. Try với symbol khác

---

## Bước 3: Test FVG Với Real Data

### 3.1. Chạy test script

```bash
python test_fvg_real_data.py
```

### 3.2. Kết quả

**Console output:**
```
============================================================
LOADING REAL MT5 DATA
============================================================
✓ File found: data/EURUSD_M15_30days.csv
✓ Data loaded
   Rows: 2880
   Date range: 2024-09-24 to 2024-10-24
   Duration: 30 days

============================================================
TESTING FVG MODULES WITH REAL DATA
============================================================
✓ ATR calculated
   Mean ATR: 0.00045

✓ Manager initialized
   Lookback: 90 days
   Min gap ATR ratio: 0.3

Processing candles...
   [10%] Candle 288/2880: 5 active FVGs
   [20%] Candle 576/2880: 8 active FVGs
   ...
   [100%] Candle 2880/2880: 12 active FVGs

FVG Statistics:
   Created:
      Bullish: 145
      Bearish: 138
      Total: 283

   Status:
      Active: 12
      Touched: 271

   Touch Rate:
      Bullish: 91.7%
      Bearish: 92.8%
      Overall: 92.3%

Current Market Structure:
   Current Price: 1.08234
   Market Bias: BOTH_FVG
   Active FVGs: 12
   Bullish FVGs below: 6
   Bearish FVGs above: 6

   📍 Nearest Bullish Target:
      Range: 1.08150 - 1.08180
      Distance: 0.00054 (5.4 pips)
      Created: 2024-10-23 14:30:00
      Age: 1.2 days

   📍 Nearest Bearish Target:
      Range: 1.08290 - 1.08320
      Distance: 0.00056 (5.6 pips)
      Created: 2024-10-23 16:15:00
      Age: 0.8 days
```

### 3.3. Output files

**Charts (HTML):**
```
logs/charts/real_data/
  ├── fvg_chart_20241024_HHMMSS.html         ← Main chart
  └── fvg_statistics_20241024_HHMMSS.html    ← Statistics
```

**Data (CSV):**
```
logs/
  ├── fvg_real_data_history.csv    ← All FVGs (283 records)
  └── fvg_real_data_active.csv     ← Active FVGs only (12 records)
```

---

## Bước 4: Phân Tích Kết Quả

### 4.1. Xem charts

**Mở trong browser:**
```bash
# Linux/Mac
xdg-open logs/charts/real_data/fvg_chart_*.html

# Windows
start logs\charts\real_data\fvg_chart_*.html
```

**Những gì cần check:**
- ✅ FVG zones có hợp lý không
- ✅ Bullish FVG (green) ở dưới giá
- ✅ Bearish FVG (red) ở trên giá
- ✅ FVG touched (gray) được mark đúng
- ✅ Gap size có đủ lớn visible với naked eye

### 4.2. Phân tích CSV

**Load data:**
```python
import pandas as pd

# Load history
history = pd.read_csv('logs/fvg_real_data_history.csv')
print(f"Total FVGs: {len(history)}")

# Filter active FVGs
active = history[history['is_active'] == True]
print(f"Active: {len(active)}")

# Filter touched FVGs
touched = history[history['is_touched'] == True]
print(f"Touched: {len(touched)}")

# Statistics
print("\nFVG Statistics:")
print(history[['fvg_type', 'gap_size', 'strength']].describe())
```

**Analyze patterns:**
```python
# Touch time analysis
touched['touch_time'] = pd.to_datetime(touched['touched_timestamp']) - pd.to_datetime(touched['created_timestamp'])
touched['touch_time_hours'] = touched['touch_time'].dt.total_seconds() / 3600

print("\nTouch Time Statistics:")
print(f"  Mean: {touched['touch_time_hours'].mean():.1f} hours")
print(f"  Median: {touched['touch_time_hours'].median():.1f} hours")
print(f"  Min: {touched['touch_time_hours'].min():.1f} hours")
print(f"  Max: {touched['touch_time_hours'].max():.1f} hours")
```

### 4.3. Verify Market Structure

**Check bias accuracy:**
```python
from config import get_data_filepath
from core.fvg import FVGManager
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv(get_data_filepath(), index_col='time', parse_dates=True)

# Calculate ATR
def calc_atr(data, period=14):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

atr = calc_atr(data)

# Initialize manager
manager = FVGManager()

# Process data
for i in range(20, len(data)):
    manager.update(data.iloc[:i+1], i, atr.iloc[i])

# Check current structure
structure = manager.get_market_structure(data['close'].iloc[-1])

print(f"\nMarket Structure:")
print(f"  Bias: {structure['bias']}")
print(f"  Active FVGs: {structure['total_active_fvgs']}")

# Validate
if structure['bias'] == 'BULLISH_BIAS':
    print("  ✓ Only trade BUY signals")
elif structure['bias'] == 'BEARISH_BIAS':
    print("  ✓ Only trade SELL signals")
elif structure['bias'] == 'BOTH_FVG':
    print("  ✓ Trade both directions based on indicators")
else:
    print("  ✗ NO TRADE - No FVG targets")
```

---

## Bước 5: So Sánh Synthetic vs Real Data

### 5.1. Test với cả 2 loại data

```bash
# Test synthetic data
python test_fvg_complete.py

# Test real data
python test_fvg_real_data.py
```

### 5.2. So sánh metrics

| Metric | Synthetic Data | Real Data |
|--------|---------------|-----------|
| Total FVGs | 164 | 283 |
| Bullish | 88 (53.7%) | 145 (51.2%) |
| Bearish | 76 (46.3%) | 138 (48.8%) |
| Touch Rate | 100% | 92.3% |
| Active FVGs | 0 | 12 |
| Avg Strength | 0.72 | 0.68 |

**Observations:**
- ✅ Real data có nhiều FVGs hơn (thị trường thật biến động hơn)
- ✅ Touch rate thấp hơn (có FVG chưa touched)
- ✅ Distribution bullish/bearish tương tự (~50/50)

---

## Bước 6: Download Nhiều Symbols/Timeframes

### 6.1. Script tự động

Tạo file `download_multiple.py`:

```python
#!/usr/bin/env python3
"""Download multiple symbols/timeframes"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.download_mt5_data import MT5DataDownloader

# Config
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
TIMEFRAMES = ['M15', 'H1', 'H4']
DAYS = 30

def main():
    downloader = MT5DataDownloader()

    if not downloader.connect():
        return 1

    try:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                print(f"\n{'='*60}")
                print(f"Downloading {symbol} {tf}...")
                print(f"{'='*60}")

                data = downloader.download_data(
                    symbol=symbol,
                    timeframe_name=tf,
                    days=DAYS,
                    save_csv=True
                )

                if data is not None:
                    print(f"✓ {symbol} {tf} downloaded")
                else:
                    print(f"✗ {symbol} {tf} failed")

    finally:
        downloader.disconnect()

    return 0

if __name__ == '__main__':
    exit(main())
```

**Chạy:**
```bash
python download_multiple.py
```

---

## Troubleshooting Common Issues

### Issue 1: MT5 not found
```
❌ Failed to initialize MT5
```
**Fix:**
1. Check `MT5_CONFIG['path']` in config.py
2. Verify MT5 installed
3. Try `'path': None` for auto-detect

### Issue 2: No permission
```
❌ Symbol EURUSD not found
```
**Fix:**
1. Open MT5 → Market Watch
2. Right click → Show All
3. Find symbol and enable

### Issue 3: Old data
```
❌ No data received
```
**Fix:**
1. Reduce `days` parameter (try 7 days first)
2. Check broker provides historical data
3. Verify account type (demo/real)

### Issue 4: File not found when testing
```
❌ File not found: data/EURUSD_M15_30days.csv
```
**Fix:**
```bash
# Download data first
python data/download_mt5_data.py

# Then test
python test_fvg_real_data.py
```

---

## Best Practices

### 1. Data Management
- ✅ Download fresh data regularly (weekly)
- ✅ Keep backups of important datasets
- ✅ Use consistent timeframes for analysis

### 2. FVG Analysis
- ✅ Test với ít nhất 30 days data
- ✅ Compare multiple symbols
- ✅ Analyze touch rates per symbol/timeframe

### 3. Performance
- ✅ Start với 30 days (faster testing)
- ✅ Expand to 90 days for production
- ✅ Use H1/H4 for faster backtests

---

## Quick Reference

```bash
# 1. Configure MT5 path
nano config.py  # Edit MT5_CONFIG['path']

# 2. Download data
python data/download_mt5_data.py

# 3. Test FVG
python test_fvg_real_data.py

# 4. View charts
xdg-open logs/charts/real_data/*.html

# 5. Analyze CSV
head logs/fvg_real_data_history.csv
```

---

## Next Steps

Sau khi test xong với real data:
1. ✅ Verify FVG detection accuracy
2. ✅ Analyze touch rates per symbol
3. ✅ Integrate with indicators module
4. ✅ Develop full trading strategy
5. ✅ Backtest with FVG + confluence signals
6. ✅ Paper trading (virtual mode)
7. ✅ Live trading (real mode)
