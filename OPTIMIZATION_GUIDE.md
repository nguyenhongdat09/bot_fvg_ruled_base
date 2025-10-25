# Strategy Optimization Guide

## Quick Start

Bạn có win rate thấp (< 40%)? Hãy optimize strategy!

### Step 1: Run Optimization Script

```bash
python examples/optimize_strategy.py
```

Chọn optimization mode:
- `1` = Confidence & Filters (RECOMMENDED FIRST) - Tìm best filtering thresholds
- `2` = SL/TP Ratios - Tìm best risk/reward ratio
- `3` = Confluence Weights - Tìm best indicator weighting
- `all` = Run tất cả (20-30 phút)

### Step 2: Check Results

Results được save vào `data/` folder:
- `optimization_confidence_TIMESTAMP.csv` - Kết quả test confidence & filters
- `optimization_sltp_TIMESTAMP.csv` - Kết quả test SL/TP ratios
- `optimization_weights_TIMESTAMP.csv` - Kết quả test confluence weights

Top 10 best configurations sẽ được print ra console.

### Step 3: Apply Best Settings

Mở `config.py` và update `BACKTEST_CONFIG` với best parameters từ optimization results.

Hoặc dùng pre-configured optimized settings:

```python
# In examples/run_backtest.py, change line 28:
from config import DATA_DIR, BACKTEST_CONFIG_OPTIMIZED as BACKTEST_CONFIG
```

### Step 4: Re-run Backtest

```bash
python examples/run_backtest.py
```

Verify win rate đã improve!

---

## Understanding the Results

### Composite Score

Optimization script dùng **composite score** để rank configurations:

```
Composite Score = (Win Rate × Profit Factor × (1 + Return%/100)) / (1 + Max DD%/100)
```

**Higher is better!**

- Rewards: High win rate, high profit factor, high return
- Penalizes: High drawdown

### Key Metrics to Look For

✅ **Good Configuration:**
- Win Rate: 45-55%
- Profit Factor: > 1.5
- Return: > 5% over 180 days
- Max Drawdown: < 10%
- Reasonable trade count (100-200 trades)

❌ **Bad Configuration:**
- Win Rate: < 35%
- Profit Factor: < 1.2
- Max Drawdown: > 15%
- Too many trades (> 300) = overtrading
- Too few trades (< 50) = too selective

---

## Pre-configured Settings Comparison

### BACKTEST_CONFIG (Default)

**Profile:** Conservative testing baseline

```python
min_confidence: 70%
adx_threshold: 25
consecutive_losses_trigger: 3
SL/TP: 1.5x/3.0x ATR (1:2 R:R)
```

**Expected Results:**
- Win Rate: 35-40%
- Trades: 250-300
- Return: 2-5%
- Use this as BASELINE for comparison

---

### BACKTEST_CONFIG_OPTIMIZED (Recommended)

**Profile:** Optimized for better win rate

```python
min_confidence: 85%  ← More selective
adx_threshold: 30    ← Stronger trends only
consecutive_losses_trigger: 5  ← Safer martingale
SL/TP: 2.0x/4.0x ATR (1:2 R:R)  ← Better risk/reward
```

**Expected Results:**
- Win Rate: 45-50%
- Trades: 100-150
- Return: 5-10%
- **Use this for REAL trading**

**Key Improvements:**
- 15% higher confidence threshold → fewer low-quality trades
- 5 point higher ADX → only strong trends
- Safer martingale (5 losses vs 3) → less risk
- Better SL/TP ratio → wider SL, higher TP

---

### BACKTEST_CONFIG_AGGRESSIVE (High Risk)

**Profile:** High risk, high reward

```python
min_confidence: 60%  ← Very loose
adx_threshold: N/A   ← No filter (trades everything)
consecutive_losses_trigger: 1  ← DANGEROUS!
SL/TP: 1.0x/5.0x ATR (1:5 R:R)  ← Tight SL, high TP
max_concurrent_trades: 3  ← Multiple positions
```

**Expected Results:**
- Win Rate: 30-35% (LOW!)
- Trades: 400-500 (OVERTRADING!)
- Return: -5% to +15% (VOLATILE!)
- Max Drawdown: 20-30% (DANGEROUS!)

**WARNING:**
- ⚠️ Only for experienced traders
- ⚠️ Requires large account ($5000+)
- ⚠️ High chance of account blow-up
- ⚠️ NOT recommended for beginners

---

## Optimization Strategy

### Phase 1: Find Best Filters (Most Important)

Run optimization 1 first:
```bash
python examples/optimize_strategy.py
> 1
```

This finds optimal:
- `min_confidence_score` (70-90%)
- `adx_threshold` (20-35)
- `consecutive_losses_trigger` (1-5)

**Goal:** Maximize win rate by filtering bad trades

**Time:** 10-15 minutes

**Look for:**
- Win rate improvement of 5-10%
- Reasonable trade count (100-200)
- Lower drawdown

---

### Phase 2: Optimize Risk/Reward

After finding best filters, run optimization 2:
```bash
python examples/optimize_strategy.py
> 2
```

This finds optimal:
- `atr_sl_multiplier` (1.0-2.5)
- `atr_tp_multiplier` (2.0-5.0)

**Goal:** Maximize profit factor with better R:R ratio

**Time:** 5-10 minutes

**Look for:**
- Profit factor > 1.5
- Balance between win rate and R:R
- Avg Win > 2× Avg Loss

---

### Phase 3: Fine-tune Indicators (Optional)

Finally, run optimization 3:
```bash
python examples/optimize_strategy.py
> 3
```

This tests different confluence weight combinations.

**Goal:** Find best indicator weighting

**Time:** 5-10 minutes

**Look for:**
- Small improvements (1-2% win rate)
- Consistency across different weight combinations
- Higher FVG weight usually better (primary signal)

---

## Example Workflow

### Scenario: Win Rate 36.7% (Too Low!)

**Current Config:**
```python
min_confidence: 70%
adx_threshold: 25
consecutive_losses_trigger: 1  # DANGEROUS!
```

**Step 1:** Run optimization 1
```bash
python examples/optimize_strategy.py
> 1
```

**Results:**
```
Top Result:
  Confidence: 85%, ADX: 30, Trigger: 3
  Win Rate: 47.2%
  Profit Factor: 1.68
  Return: 8.3%
```

**Step 2:** Update config.py
```python
BACKTEST_CONFIG = {
    ...
    'min_confidence_score': 85.0,  # Changed from 70
    'adx_threshold': 30.0,         # Changed from 25
    'consecutive_losses_trigger': 3, # Changed from 1
    ...
}
```

**Step 3:** Re-run backtest
```bash
python examples/run_backtest.py
```

**New Results:**
- Win Rate: 47.2% ✅ (was 36.7%)
- Trades: 142 (was 281) - less but higher quality
- Return: 8.3% ✅ (was 2.8%)

**Success!** Win rate improved by 10.5% just by better filtering!

---

## Tips for Best Results

### 1. Start with Confidence & Filters
This optimization gives the BIGGEST improvement (5-10% win rate boost).
Always run this first!

### 2. Don't Overtrade
If you see 300+ trades in 180 days, confidence threshold is too low.
Increase `min_confidence_score` by 5-10%.

### 3. Watch Martingale Trigger
- `trigger: 1` = VERY DANGEROUS (63% of trades in martingale)
- `trigger: 3` = Balanced (30-40% martingale)
- `trigger: 5` = Conservative (10-20% martingale)

### 4. Risk/Reward Ratio
- 1:1.5 = Too tight, need 60%+ win rate
- 1:2 = Balanced, need 40%+ win rate ✅
- 1:3 = Good, need 35%+ win rate ✅
- 1:5 = Aggressive, need 25%+ win rate (but high variance)

### 5. Test Multiple Symbols
Don't optimize on just GBPUSD!
Test on EURUSD, USDJPY, AUDUSD to verify settings work across symbols.

### 6. Avoid Overfitting
If config works great on one symbol but terrible on others = overfitting.
Look for settings that are consistently good across multiple symbols.

---

## Troubleshooting

### Problem: All results have win rate < 30%

**Possible causes:**
1. Strategy doesn't work on this symbol/timeframe
2. Data quality issues
3. FVG detection not working properly

**Solution:**
- Try different symbol (EURUSD instead of GBPUSD)
- Try different timeframe (H1 base instead of M15)
- Check FVG detection: `python -m core.fvg.fvg_detector --test`

---

### Problem: Optimization too slow

**Solution:**
- Run smaller grid (edit optimize_strategy.py)
- Test fewer parameter combinations
- Use faster computer
- Run overnight

---

### Problem: Best config has < 50 trades

**Possible causes:**
- Confidence threshold too high (> 90%)
- ADX threshold too high (> 35)
- Not enough data (need more days)

**Solution:**
- Lower confidence by 5% (85 → 80)
- Lower ADX by 5 points (30 → 25)
- Download more data (180 → 360 days)

---

## Next Steps After Optimization

1. ✅ **Found best parameters** → Update config.py
2. ✅ **Verified on backtest** → Win rate improved
3. 📊 **Test on other symbols** → EURUSD, USDJPY, AUDUSD
4. 📊 **Test on other timeframes** → M15, M30, H1
5. 📊 **Walk-forward validation** → Split data, test on future data
6. 🚀 **Paper trade** → Test on live data (no real money)
7. 🚀 **Live trade** → Start with small account

---

## Summary

### Quick Wins for Better Win Rate:

1. **Increase min_confidence_score**: 70 → 85 (+10-15% win rate)
2. **Increase adx_threshold**: 25 → 30 (+5% win rate)
3. **Fix martingale trigger**: 1 → 3 or 5 (safer)
4. **Better SL/TP ratio**: 1.5/3.0 → 2.0/4.0 (better R:R)

### Expected Improvements:

| Change | Win Rate | Trades | Return |
|--------|----------|--------|--------|
| Default Config | 35-40% | 250-300 | 2-5% |
| After Optimization | 45-50% | 100-150 | 5-10% |
| Improvement | +10% | -50% | +5% |

**Trade Quality > Trade Quantity**

Fewer high-quality trades beat many low-quality trades every time!

---

Good luck với optimization! 🚀
