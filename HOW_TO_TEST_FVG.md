# Hướng Dẫn Test FVG Modules

## Tổng Quan

Có 3 cách test FVG modules theo mức độ từ đơn giản đến chi tiết:

1. **Test nhanh** - `test_fvg_simple.py`
2. **Test chi tiết** - `test_fvg_interactive.py`
3. **Test toàn diện** - `test_fvg_complete.py` (có visualization)

---

## 1. Test Nhanh (Recommended đầu tiên)

### Chạy:
```bash
python test_fvg_simple.py
```

### Kết quả:
- Tạo 100 nến test data
- Phát hiện ~20-30 FVGs
- Test FVG Manager
- Test Market Structure
- Export CSV

### Output:
```
✓ 22 FVGs detected
  - Bullish: 14
  - Bearish: 8
  - Avg Strength: 0.67
✓ Export: logs/fvg_simple_test.csv
```

### Kiểm tra:
```bash
# Xem CSV file
cat logs/fvg_simple_test.csv | head -5
```

---

## 2. Test Chi Tiết (Recommended để hiểu logic)

### Chạy:
```bash
python test_fvg_interactive.py
```

### Các test cases:

#### Test 1: FVG Object Creation
- Tạo FVG object
- Test touching logic
- Test valid target logic

#### Test 2: FVG Detection
- Phát hiện FVG từ OHLC data
- Show 3 nến tạo gap
- Validate gap size

#### Test 3: Market Structure Analysis
- Test 3 price levels khác nhau
- Show bias (BULLISH_BIAS, BEARISH_BIAS, BOTH_FVG, NO_FVG)
- Test signal validation

#### Test 4: FVG Tracking Over Time
- Process 50 nến
- Track FVG creation/touching
- Show statistics

### Output mẫu:
```
>>> Current Price: 1.10300
  Market Bias: BOTH_FVG
  Bullish FVGs below: 1
  Bearish FVGs above: 1
  Nearest Bullish: 1.10100-1.10200 (distance: 0.00100)
  Nearest Bearish: 1.10500-1.10600 (distance: 0.00200)
  Signal validation:
    BUY: valid=True, has_target=True
    SELL: valid=True, has_target=True
```

---

## 3. Test Toàn Diện (Có Visualization)

### Chạy:
```bash
python test_fvg_complete.py
```

### Kết quả:
- Tạo 500 nến test data
- Phát hiện ~150-200 FVGs
- Test tất cả modules
- **Tạo HTML charts**
- Export CSV đầy đủ

### Output files:
```
📊 Charts (HTML):
  logs/charts/test_fvg_chart.html         (Main chart)
  logs/charts/test_fvg_statistics.html    (Statistics)

📋 Data (CSV):
  logs/fvg_history_test.csv               (All FVGs)
  logs/fvg_active_test.csv                (Active FVGs)
```

### Xem charts:

**Option 1: Mở trong browser**
```bash
# Linux/Mac
xdg-open logs/charts/test_fvg_chart.html

# Windows
start logs\charts\test_fvg_chart.html
```

**Option 2: Copy đường dẫn đầy đủ**
```bash
realpath logs/charts/test_fvg_chart.html
# Copy path và paste vào browser
```

**Option 3: Nếu có web server**
```bash
cd logs/charts
python -m http.server 8000
# Mở browser: http://localhost:8000/test_fvg_chart.html
```

---

## 4. Test Với Data Thật (MT5)

### Bước 1: Download data từ MT5
```python
# data/download_mt5_data.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# Initialize MT5
mt5.initialize()

# Download data
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_M15
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)

# Save
data.to_csv('data/EURUSD_M15_real.csv')
print(f"Downloaded {len(data)} candles")

mt5.shutdown()
```

### Bước 2: Test với data thật
```python
# test_fvg_real_data.py
import pandas as pd
from core.fvg import FVGManager
from core.indicators.volatility import calculate_atr

# Load real data
data = pd.read_csv('data/EURUSD_M15_real.csv', index_col='time', parse_dates=True)

# Calculate ATR
atr = calculate_atr(data, period=14)

# Initialize manager
manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.3)

# Process all candles
for i in range(20, len(data)):
    manager.update(data.iloc[:i+1], i, atr.iloc[i])

# Get statistics
stats = manager.get_statistics()
print(f"Real data FVG statistics:")
print(f"  Total created: {stats['total_bullish_created'] + stats['total_bearish_created']}")
print(f"  Active: {stats['total_active']}")
print(f"  Touch rate: {(stats['bullish_touch_rate'] + stats['bearish_touch_rate']) / 2:.1f}%")

# Visualize
from core.fvg import FVGVisualizer
visualizer = FVGVisualizer()
visualizer.create_fvg_report(
    data=data,
    fvgs=manager.all_fvgs_history,
    output_dir='logs/charts/real_data'
)
```

---

## 5. Verify Kết Quả

### Check 1: FVG detection logic
Mở chart HTML và verify:
- ✅ Bullish FVG (green zone) ở dưới giá hiện tại
- ✅ Bearish FVG (red zone) ở trên giá hiện tại
- ✅ Gray zones là FVG đã touched
- ✅ Gap có đủ lớn (visible với naked eye)

### Check 2: Touching logic
Kiểm tra trong CSV:
```bash
# Count touched FVGs
cat logs/fvg_history_test.csv | grep -c "True"
```

Verify trong chart:
- FVG chuyển từ màu → gray ngay khi price chạm

### Check 3: Market Structure
Test với interactive script:
```bash
python test_fvg_interactive.py | grep "Market Bias"
```

Verify logic:
- Price giữa 2 FVG → BOTH_FVG
- Chỉ có FVG dưới → BULLISH_BIAS
- Chỉ có FVG trên → BEARISH_BIAS
- Không có FVG → NO_FVG

### Check 4: Statistics
```bash
python test_fvg_simple.py | grep "Avg Strength"
```

Expected values:
- Avg Strength: 0.5 - 1.0 (normal)
- Touch rate: 70-100% (high trong test data ngắn)
- Bullish/Bearish ratio: ~50/50 (trong random data)

---

## 6. Common Issues & Debug

### Issue 1: Không phát hiện FVG
```python
# Check gap size threshold
detector = FVGDetector(min_gap_atr_ratio=0.2)  # Giảm threshold
```

### Issue 2: Tất cả FVG bị touched
```python
# Normal với test data ngắn
# Test với data dài hơn:
data = create_test_data(n=1000)  # Tăng từ 100 → 1000
```

### Issue 3: Chart không hiển thị
```bash
# Check plotly installed
pip install plotly

# Check file size
ls -lh logs/charts/*.html
# Should be ~5MB
```

### Issue 4: CSV rỗng
```python
# Check manager has data
print(f"Total FVGs: {len(manager.all_fvgs_history)}")

# Export manually
df = manager.export_history_to_dataframe()
print(df.head())
```

---

## 7. Test Checklist

Để đảm bảo FVG modules hoạt động đúng:

- [ ] Test 1: FVG object creation ✓
- [ ] Test 2: FVG detection from OHLC ✓
- [ ] Test 3: FVG touching logic ✓
- [ ] Test 4: Market structure analysis ✓
- [ ] Test 5: FVG tracking over time ✓
- [ ] Test 6: Data export (CSV) ✓
- [ ] Test 7: Visualization (HTML) ✓
- [ ] Test 8: Signal validation ✓
- [ ] Test 9: Real data (MT5) ⏳
- [ ] Test 10: Backtest integration ⏳

---

## 8. Next Steps

Sau khi test xong FVG modules:

1. **Review charts** - Xem visualization có đúng logic không
2. **Verify statistics** - Check touch rate, gap size có hợp lý
3. **Test với data thật** - Download từ MT5 và test
4. **Tích hợp strategy** - Kết hợp với indicators
5. **Backtest** - Chạy backtest với FVG signals

---

## Quick Commands

```bash
# Test nhanh
python test_fvg_simple.py

# Test chi tiết
python test_fvg_interactive.py

# Test full + charts
python test_fvg_complete.py

# Xem charts
xdg-open logs/charts/test_fvg_chart.html

# Xem CSV
head -20 logs/fvg_history_test.csv

# Count FVGs
wc -l logs/fvg_history_test.csv

# Find all test files
find . -name "test_fvg*.py"
```

---

## Support

Nếu gặp lỗi:
1. Check dependencies: `pip install -r requirements.txt`
2. Check Python version: `python --version` (need 3.10+)
3. Run với verbose: `python -v test_fvg_simple.py`
4. Check logs: `ls -la logs/`
