# HƯỚNG DẪN DOWNLOAD DATA TỪ MT5

## 📖 Tổng quan

Có **2 cách** download data từ MetaTrader 5:

### **1. Single Download** - Download từng cặp tiền
- File: `data/download_mt5_data.py`
- Dùng khi: Download 1 symbol + 1 timeframe
- Bất tiện khi test nhiều cặp tiền

### **2. Batch Download** - Download hàng loạt ⭐ KHUYẾN NGHỊ
- File: `data/batch_download_mt5_data.py`
- Dùng khi: Download NHIỀU symbols + NHIỀU timeframes cùng lúc
- **Chạy 1 lần, xong hết!**

---

## 🚀 BATCH DOWNLOAD (Khuyến nghị)

### **Bước 1: Config symbols và timeframes**

Mở file **`config.py`**, tìm section **`BATCH_DOWNLOAD_CONFIG`** (dòng ~70):

```python
BATCH_DOWNLOAD_CONFIG = {
    # Symbols to download
    'symbols': [
        'EURUSD',      # ← Chọn các cặp tiền muốn download
        'GBPUSD',
        'USDJPY',
        'AUDUSD',
        'USDCAD',
        'NZDUSD',
        'USDCHF',
        # 'XAUUSD',    # Uncomment để thêm Gold
    ],

    # Timeframes
    'timeframes': [
        'M15',         # ← Chọn các timeframes muốn download
        'H1',
        'H4',
        # 'M5',        # Uncomment nếu cần
        # 'D1',        # Uncomment nếu cần
    ],

    # Data range
    'days': 180,               # Download 180 ngày gần nhất

    # Settings
    'skip_existing': True,     # Bỏ qua file đã tồn tại
    'delay_between': 0.5,      # Delay 0.5s giữa mỗi download
}
```

**Ví dụ - Bạn muốn download:**
- 7 cặp tiền: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF
- 3 timeframes: M15, H1, H4
- 180 ngày data

**→ Tổng: 7 × 3 = 21 files sẽ được download!**

---

### **Bước 2: Chạy batch download**

```bash
# Windows
python data\batch_download_mt5_data.py

# Linux/Mac
python data/batch_download_mt5_data.py
```

---

### **Bước 3: Xem kết quả**

```
================================
BATCH DOWNLOAD MT5 DATA
================================

📊 Configuration:
   Symbols: 7 (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD...)
   Timeframes: 3 (M15, H1, H4)
   Days: 180
   Total downloads: 21
   Skip existing: True

================================
CONNECTING TO METATRADER 5
================================
✓ Connected to MT5

================================
DOWNLOADING...
================================

[1/21] EURUSD M15
------------------------------------------------------------
✅ Success: 17280 candles -> EURUSD_M15_180days.csv

[2/21] EURUSD H1
------------------------------------------------------------
✅ Success: 4320 candles -> EURUSD_H1_180days.csv

[3/21] EURUSD H4
------------------------------------------------------------
✅ Success: 1080 candles -> EURUSD_H4_180days.csv

...

[21/21] USDCHF H4
------------------------------------------------------------
✅ Success: 1080 candles -> USDCHF_H4_180days.csv

================================
DOWNLOAD SUMMARY
================================

✅ Successful: 21
   EURUSD   M15  -> EURUSD_M15_180days.csv
   EURUSD   H1   -> EURUSD_H1_180days.csv
   EURUSD   H4   -> EURUSD_H4_180days.csv
   GBPUSD   M15  -> GBPUSD_M15_180days.csv
   ...

⏭️  Skipped: 0
❌ Failed: 0

📁 Files saved to: data/

📊 Success Rate: 100.0% (21/21)
```

---

### **Bước 4: Kiểm tra files**

```bash
# List all downloaded files
ls data/*.csv

# Hoặc Windows
dir data\*.csv
```

**Output:**
```
data/EURUSD_M15_180days.csv
data/EURUSD_H1_180days.csv
data/EURUSD_H4_180days.csv
data/GBPUSD_M15_180days.csv
data/GBPUSD_H1_180days.csv
...
```

---

## 📋 Config Options

### **1. Chọn symbols**

```python
'symbols': [
    # Major pairs (7 cặp chính)
    'EURUSD',
    'GBPUSD',
    'USDJPY',
    'AUDUSD',
    'USDCAD',
    'NZDUSD',
    'USDCHF',

    # Cross pairs
    'EURJPY',
    'GBPJPY',
    'EURGBP',

    # Commodities
    'XAUUSD',  # Gold
    'XAGUSD',  # Silver
    'USOIL',   # Oil

    # Indices
    'US30',    # Dow Jones
    'NAS100',  # Nasdaq
    'SPX500',  # S&P 500
],
```

**Lưu ý:** Tên symbol phụ thuộc vào broker. Một số brokers dùng:
- `XAUUSD` → `GOLD`
- `USOIL` → `WTI` hoặc `CRUDE`
- `US30` → `DJ30` hoặc `US30Cash`

Kiểm tra tên symbol trong MT5 của bạn!

---

### **2. Chọn timeframes**

```python
'timeframes': [
    'M1',    # 1 minute
    'M5',    # 5 minutes
    'M15',   # 15 minutes
    'M30',   # 30 minutes
    'H1',    # 1 hour
    'H4',    # 4 hours
    'D1',    # Daily
],
```

**Khuyến nghị:** Chỉ download timeframes cần dùng để tiết kiệm thời gian:
- Strategy M15 → Download: M15, H1, H4
- Strategy H1 → Download: H1, H4, D1

---

### **3. Data range**

```python
'days': 180,  # Download 180 ngày gần nhất
```

**Lưu ý:**
- Nhiều data = test chính xác hơn
- Nhưng download lâu hơn
- Khuyến nghị: 180 ngày (6 tháng) cho test

---

### **4. Skip existing**

```python
'skip_existing': True,  # Bỏ qua file đã tồn tại
```

**True:** Không download lại file đã có → nhanh!
**False:** Download lại tất cả → dùng khi muốn update data mới

---

### **5. Error handling**

```python
'continue_on_error': True,  # Tiếp tục nếu 1 symbol lỗi
'max_retries': 2,           # Retry 2 lần nếu lỗi
```

---

## 📊 Use Cases

### **Use Case 1: Download tất cả majors + 3 timeframes**

```python
BATCH_DOWNLOAD_CONFIG = {
    'symbols': [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
        'USDCAD', 'NZDUSD', 'USDCHF'
    ],
    'timeframes': ['M15', 'H1', 'H4'],
    'days': 180,
}
# → 7 × 3 = 21 files
```

---

### **Use Case 2: Download chỉ 2 cặp tiền để test nhanh**

```python
BATCH_DOWNLOAD_CONFIG = {
    'symbols': ['EURUSD', 'GBPUSD'],  # Chỉ 2 cặp
    'timeframes': ['M15', 'H1'],       # Chỉ 2 timeframes
    'days': 90,                         # 3 tháng data
}
# → 2 × 2 = 4 files
```

---

### **Use Case 3: Download Gold + Oil + Indices**

```python
BATCH_DOWNLOAD_CONFIG = {
    'symbols': [
        'XAUUSD',   # Gold
        'USOIL',    # Oil
        'US30',     # Dow Jones
        'NAS100',   # Nasdaq
    ],
    'timeframes': ['H1', 'H4', 'D1'],
    'days': 365,  # 1 năm data
}
# → 4 × 3 = 12 files
```

---

### **Use Case 4: Update data mới (download lại)**

```python
BATCH_DOWNLOAD_CONFIG = {
    'symbols': ['EURUSD', 'GBPUSD'],
    'timeframes': ['M15'],
    'days': 30,  # Chỉ 30 ngày gần nhất
    'skip_existing': False,  # ← Download lại dù file đã tồn tại
}
```

---

## ⚠️ Troubleshooting

### **Lỗi: Symbol not found**

```
❌ Symbol XAUUSD not found
```

**Nguyên nhân:** Broker của bạn dùng tên khác

**Giải pháp:**
1. Mở MT5
2. Market Watch → Right click → Show All
3. Tìm Gold → Xem tên chính xác (có thể là `GOLD`, `XAU/USD`, `XAUUSD`)
4. Dùng tên đó trong config

---

### **Lỗi: Failed to connect to MT5**

```
❌ Failed to initialize MT5
```

**Giải pháp:**
1. Mở MT5 trước khi chạy script
2. Đảm bảo MT5 đang chạy
3. Kiểm tra MT5 path trong `config.py` → `MT5_CONFIG['path']`

---

### **Lỗi: No data received**

```
❌ No data received
```

**Nguyên nhân:**
- Symbol không có data cho timeframe đó
- Hoặc không có historical data
- Hoặc MT5 chưa login

**Giải pháp:**
- Login vào MT5 account
- Thử download timeframe khác
- Giảm số ngày (từ 180 → 90)

---

## 🔄 So sánh: Single vs Batch Download

| Tiêu chí | Single Download | Batch Download ⭐ |
|----------|-----------------|-------------------|
| **Files** | 1 symbol × 1 TF | Nhiều symbols × Nhiều TFs |
| **Command** | `python data/download_mt5_data.py` | `python data/batch_download_mt5_data.py` |
| **Config** | Edit code hoặc CLI args | Chỉ edit `config.py` |
| **Tiện lợi** | ❌ Phải chạy nhiều lần | ✅ Chạy 1 lần xong hết |
| **Skip existing** | ❌ Không | ✅ Có |
| **Retry** | ❌ Không | ✅ Có |
| **Progress** | ✅ Có | ✅ Có + Summary |

---

## 📁 File Structure sau khi download

```
data/
├── EURUSD_M15_180days.csv
├── EURUSD_H1_180days.csv
├── EURUSD_H4_180days.csv
├── GBPUSD_M15_180days.csv
├── GBPUSD_H1_180days.csv
├── GBPUSD_H4_180days.csv
├── USDJPY_M15_180days.csv
├── USDJPY_H1_180days.csv
├── USDJPY_H4_180days.csv
...
```

**Format:** `{SYMBOL}_{TIMEFRAME}_{DAYS}days.csv`

**Columns:** `time, open, high, low, close, volume`

---

## 🎯 Quick Start

### **Cách nhanh nhất:**

```bash
# 1. Mở config.py
notepad config.py  # Windows
code config.py     # VS Code

# 2. Tìm BATCH_DOWNLOAD_CONFIG (dòng ~70)
# 3. Chỉnh symbols và timeframes
# 4. Save (Ctrl+S)

# 5. Chạy batch download
python data/batch_download_mt5_data.py

# 6. Đợi download xong (có thể 5-15 phút)
# 7. Kiểm tra files
dir data\*.csv     # Windows
ls data/*.csv      # Linux/Mac
```

---

## 📝 Tóm tắt

### **Câu hỏi:**
> "Phải ngồi set config rồi chạy từng lần bất tiện nhỉ?"

### **Trả lời:**
✅ **ĐÃ FIX!** Giờ dùng **Batch Download**:

1. **Config 1 lần** trong `config.py` → `BATCH_DOWNLOAD_CONFIG`
2. **Chạy 1 lần:** `python data/batch_download_mt5_data.py`
3. **Download hết:** Tất cả symbols + timeframes

**Ví dụ:**
- 7 cặp tiền × 3 timeframes = 21 files
- Chạy 1 lần, xong hết!
- Lần sau chạy lại → tự động skip files đã tồn tại

**Không còn phải chạy từng lần nữa!**

---

**CREATED:** 2025-10-24
**STATUS:** ✅ READY TO USE
