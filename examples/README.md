# FVG Trading Bot - Backtest với SL/TP theo Pips

## Tổng quan

Hệ thống backtest cho chiến lược giao dịch FVG (Fair Value Gap) với khả năng cấu hình SL/TP theo:
- **Fixed Pips**: SL/TP cố định theo số pip
- **ATR-based**: SL/TP động theo ATR (Average True Range)

## Cài đặt

```bash
# Install dependencies
pip install pandas numpy plotly

# Run backtest
python examples/run_backtest.py
```

## Cấu hình SL/TP

Tất cả cấu hình được quản lý trong file `config.py`, phần `BACKTEST_CONFIG`.

### 1. Chế độ Fixed Pips

Sử dụng SL/TP cố định theo số pip:

```python
BACKTEST_CONFIG = {
    # ... các cấu hình khác ...
    
    # Chọn chế độ
    'sl_tp_mode': 'fixed',  # ← 'fixed' để dùng pip cố định
    
    # Cấu hình SL/TP theo pips
    'fixed_sl_pips': 20.0,  # ← Stop Loss 20 pips
    'fixed_tp_pips': 40.0,  # ← Take Profit 40 pips
}
```

**Ví dụ các cấu hình phổ biến:**
- Scalping: SL=10, TP=20
- Day Trading: SL=20, TP=40
- Swing Trading: SL=50, TP=100

### 2. Chế độ ATR-based

SL/TP tự động điều chỉnh theo volatility (ATR):

```python
BACKTEST_CONFIG = {
    # ... các cấu hình khác ...
    
    # Chọn chế độ
    'sl_tp_mode': 'atr',  # ← 'atr' để dùng ATR
    
    # Cấu hình ATR
    'atr_period': 14,         # ← Chu kỳ tính ATR (14 nến)
    'atr_sl_multiplier': 1.5, # ← SL = ATR × 1.5
    'atr_tp_multiplier': 3.0, # ← TP = ATR × 3.0
}
```

**Ví dụ các cấu hình:**
- Conservative: SL=1.0×ATR, TP=2.0×ATR
- Standard: SL=1.5×ATR, TP=3.0×ATR  
- Aggressive: SL=2.0×ATR, TP=4.0×ATR

### 3. Các cấu hình khác

```python
BACKTEST_CONFIG = {
    # === Vốn và Risk ===
    'initial_capital': 10000.0,      # Vốn ban đầu $10,000
    'risk_per_trade_percent': 1.0,   # Risk 1% mỗi lệnh
    
    # === Chi phí giao dịch ===
    'commission_per_lot': 0.0,       # Commission
    'spread_pips': 1.0,              # Spread 1 pip
    
    # === FVG Settings ===
    'fvg_lookback_days': 90,         # FVG có hiệu lực 90 ngày
    'fvg_min_gap_atr_ratio': 0.3,   # Gap tối thiểu = 0.3 × ATR
    
    # === Quản lý lệnh ===
    'max_concurrent_trades': 1,      # Tối đa 1 lệnh cùng lúc
}
```

## Sử dụng

### 1. Chạy Backtest đơn giản

```bash
python examples/run_backtest.py
```

Script này sẽ:
- Chạy backtest với cả 2 chế độ (Fixed và ATR)
- So sánh kết quả giữa 2 chế độ
- Export trades ra file CSV trong thư mục `logs/`

### 2. Test nhiều cấu hình khác nhau

```bash
python examples/test_config_variations.py
```

Script này sẽ test nhiều cấu hình SL/TP khác nhau và so sánh kết quả.

### 3. Tùy chỉnh trong code

```python
from config import BACKTEST_CONFIG
from examples.run_backtest import run_backtest

# Tạo custom config
custom_config = BACKTEST_CONFIG.copy()
custom_config['sl_tp_mode'] = 'fixed'
custom_config['fixed_sl_pips'] = 15.0
custom_config['fixed_tp_pips'] = 30.0

# Chạy backtest
backtester, metrics = run_backtest(custom_config, mode='fixed')

# Xem kết quả
print(f"Win Rate: {metrics['win_rate']:.2f}%")
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
```

## Kết quả Backtest

### Output Console

```
============================================================
BACKTEST RESULTS
============================================================

Performance Metrics:
  Total Trades: 4
  Winning Trades: 3
  Losing Trades: 1
  Win Rate: 75.00%

P&L:
  Total P&L: $348.64
  Total Return: 3.49%
  Average Win: $150.21
  Average Loss: $-102.00
  Profit Factor: 4.42

Capital:
  Initial Capital: $10,000.00
  Final Capital: $10,348.64
  Max Drawdown: $-102.00 (-0.99%)
```

### Output Files

Sau khi chạy backtest, các file sau được tạo trong thư mục `logs/`:

- `backtest_trades_fixed.csv`: Chi tiết tất cả trades ở chế độ Fixed
- `backtest_trades_atr.csv`: Chi tiết tất cả trades ở chế độ ATR

**Cấu trúc CSV:**
```csv
trade_id,signal,entry_time,entry_price,exit_time,exit_price,exit_reason,lot_size,sl_price,tp_price,pnl,pnl_pips,commission,is_open
```

## So sánh Fixed vs ATR

| Chỉ số | Fixed (20/40 pips) | ATR (1.5x/3.0x) |
|--------|-------------------|-----------------|
| **Ưu điểm** | - Đơn giản, dễ hiểu<br>- Kiểm soát risk rõ ràng<br>- Phù hợp với scalping | - Tự động điều chỉnh theo volatility<br>- Linh hoạt với điều kiện thị trường<br>- Tránh bị SL quá sớm khi biến động cao |
| **Nhược điểm** | - Không thích ứng với volatility<br>- SL có thể quá chặt hoặc quá rộng | - Phức tạp hơn<br>- SL/TP thay đổi theo thời gian<br>- Khó tính toán trước |
| **Khi nào dùng** | - Thị trường ổn định<br>- Scalping/Day trading<br>- Muốn kiểm soát chặt chẽ | - Thị trường biến động<br>- Swing trading<br>- Muốn tự động hóa |

## Cấu trúc Code

```
bot_fvg_ruled_base/
├── config.py                    # ← Cấu hình chính (BACKTEST_CONFIG)
├── core/
│   └── backtest/
│       └── backtester.py        # Engine backtest
├── examples/
│   ├── run_backtest.py          # Script chạy backtest
│   └── test_config_variations.py # Test nhiều cấu hình
└── logs/                        # Output files (CSV)
```

## Các metrics được tính

- **Total Trades**: Tổng số lệnh
- **Win Rate**: Tỷ lệ thắng (%)
- **Total Return**: Lợi nhuận tổng (%)
- **Profit Factor**: Tổng win / Tổng loss
- **Max Drawdown**: Mức sụt giảm lớn nhất
- **Average Win/Loss**: Trung bình win/loss mỗi lệnh

## Lưu ý

1. **Position Sizing**: Tự động tính dựa trên `risk_per_trade_percent`
2. **Spread**: Được áp dụng tự động cho mỗi lệnh
3. **Commission**: Cấu hình theo lot
4. **Data**: Script sử dụng sample data, có thể thay bằng data thật từ MT5

## Troubleshooting

### Không có signal nào được tạo?

Kiểm tra:
- FVG settings (`fvg_min_gap_atr_ratio`)
- Lookback period
- Data quality

### Kết quả không như mong đợi?

- Thử điều chỉnh SL/TP ratio (TP nên gấp 2-3 lần SL)
- Kiểm tra spread và commission
- Review signal generation logic

## Next Steps

1. [HOÀN THÀNH] Cấu hình SL/TP theo pips
2. [HOÀN THÀNH] Toggle Fixed/ATR mode
3. [KẾ HOẠCH] Tích hợp data thật từ MT5
4. [KẾ HOẠCH] Thêm trailing stop
5. [KẾ HOẠCH] Visualization với charts

## Liên hệ

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue trên GitHub.
