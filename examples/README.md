# Examples - FVG Trading Bot

Thư mục này chứa các ví dụ sử dụng FVG trading bot.

## 📁 Danh sách Examples

### 1. `fvg_with_indicators_example.py`
Ví dụ kết hợp FVG với indicators (RSI, Volume).

**Chức năng:**
- FVG + RSI + Volume strategy
- Verify no look-ahead bias test

**Chạy:**
```bash
python examples/fvg_with_indicators_example.py
```

---

### 2. `multi_timeframe_example.py`
Ví dụ phân tích multi-timeframe (FVG H1 + RSI M15).

**Chức năng:**
- Option 1: Manual resample & align
- Option 2: MultiTimeframeManager
- Verify no look-ahead bias test

**Chạy:**
```bash
python examples/multi_timeframe_example.py
```

---

## ⚠️ Lưu ý

### Cách chạy đúng:

**✅ ĐÚNG - Chạy từ root directory:**
```bash
# Đảm bảo bạn đang ở root directory (nơi có folder core/, data/, examples/)
cd E:\Bot_FVG\trading_bot

# Chạy script
python examples/multi_timeframe_example.py
```

**❌ SAI - Chạy từ examples/ directory:**
```bash
cd examples
python multi_timeframe_example.py  # ❌ SẼ BỊ LỖI!
```

### Yêu cầu:

1. **Data file:** Đảm bảo có file `data/EURUSD_M15_30days.csv`
   - Nếu chưa có, chạy: `python data/download_mt5_data.py`

2. **Thư viện:**
   ```bash
   pip install pandas pandas_ta MetaTrader5
   ```

---

## 🐛 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'core'`

**Nguyên nhân:** Script không tìm thấy module `core`

**Giải pháp:**
```bash
# Chạy từ root directory
cd E:\Bot_FVG\trading_bot
python examples/multi_timeframe_example.py
```

### Lỗi: `FileNotFoundError: data/EURUSD_M15_30days.csv`

**Nguyên nhân:** Chưa có data

**Giải pháp:**
```bash
# Download data từ MT5
python data/download_mt5_data.py
```

---

## 📖 Tài liệu tham khảo

- `HOW_TO_USE_MULTI_TIMEFRAME.md` - Hướng dẫn multi-timeframe
- `FVG_NO_LOOK_AHEAD_ANALYSIS.md` - Phân tích look-ahead bias
- `MULTI_TIMEFRAME_ANALYSIS.md` - Phân tích chi tiết

---

**CREATED:** 2025-10-24
