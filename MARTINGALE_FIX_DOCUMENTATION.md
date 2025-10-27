# Critical Fixes: SL/TP Configuration & Martingale Logic

**Date:** 2025-10-25
**Issues Fixed:**
1. SL/TP configuration confusion
2. Martingale not compounding in VIRTUAL mode

---

## ❌ Problem #1: SL/TP Configuration Not Working

### User Report:
> "dù tôi có đổi sl, tp bao nhiêu thì chạy run backtest vẫn ra pnl y chang các lần trước"

### Root Cause:

User was editing `STRATEGY_CONFIG` in `config.py`:
```python
STRATEGY_CONFIG = {
    'sl_pips': 20,
    'tp_pips': 30,
    ...
}
```

**But this config is NOT USED by backtester!** ❌

Backtester only reads from `BACKTEST_CONFIG`:
```python
# backtester.py
sl_distance = atr_value * self.config['atr_sl_multiplier']  # From BACKTEST_CONFIG!
tp_distance = atr_value * self.config['atr_tp_multiplier']  # From BACKTEST_CONFIG!
```

### Solution:

1. **Marked STRATEGY_CONFIG as LEGACY (not used)**
2. **Clarified where to edit SL/TP:**

```python
# config.py - BACKTEST_CONFIG
BACKTEST_CONFIG = {
    ...
    # ===== STOP LOSS / TAKE PROFIT (IMPORTANT: Edit these to change SL/TP!) =====
    'atr_sl_multiplier': 1.5,      # SL = ATR x 1.5 (increase for wider SL)
    'atr_tp_multiplier': 3.0,      # TP = ATR x 3.0 (increase for wider TP)
    # Example: atr_sl_multiplier=2.0, atr_tp_multiplier=4.0 for wider SL/TP
    ...
}
```

**To change SL/TP, edit these 2 parameters in BACKTEST_CONFIG!**

---

## ❌ Problem #2: Martingale Not Compounding in VIRTUAL Mode

### User Request:
> "lệnh ảo thua 3 lệnh thì vẫn phải martingale 3 lệnh này lên để lệnh thứ 4 là lần martingale thứ 4"

Translation: Virtual losses should compound lot size BEFORE switching to REAL mode.

### Old (Buggy) Logic:

```python
# Old code - WRONG!
if trade.is_loss():
    consecutive_losses += 1

    if consecutive_losses >= 3:
        if mode == VIRTUAL:
            mode = REAL
            lot_size = base_lot_size  # ❌ RESETS lot size!
        else:
            lot_size *= martingale_multiplier  # Only martingale in REAL mode
```

**Problems:**
1. Martingale ONLY happens in REAL mode
2. When switching to REAL, lot size RESETS to base ❌
3. Martingale doesn't accumulate from VIRTUAL losses

**Example with old logic:**
```
consecutive_losses_trigger = 3
base_lot_size = 0.1
martingale_multiplier = 1.3

Trade 1 (VIRTUAL): 0.1 lot, LOSS → lot stays 0.1 (no martingale!)
Trade 2 (VIRTUAL): 0.1 lot, LOSS → lot stays 0.1 (no martingale!)
Trade 3 (VIRTUAL): 0.1 lot, LOSS → lot stays 0.1 (no martingale!)
Trade 4 (REAL):    0.1 lot, LOSS → lot = 0.1 × 1.3 = 0.13 (only now!)
Trade 5 (REAL):    0.13 lot, ...
```

❌ **Lot size doesn't compound until AFTER switching to REAL!**

---

### New (Fixed) Logic:

```python
# New code - CORRECT!
if trade.is_loss():
    consecutive_losses += 1

    # Martingale on EVERY loss (VIRTUAL or REAL)
    lot_size *= martingale_multiplier
    lot_size = min(lot_size, max_lot_size)

    # Switch to REAL mode (keep already-compounded lot size)
    if consecutive_losses >= 3:
        mode = REAL
```

**Improvements:**
1. ✅ Martingale happens on EVERY loss (VIRTUAL or REAL)
2. ✅ Lot size compounds BEFORE switching to REAL
3. ✅ When switching to REAL, lot size is already at martingale level N

**Example with new logic:**
```
consecutive_losses_trigger = 3
base_lot_size = 0.1
martingale_multiplier = 1.3

Trade 1 (VIRTUAL): 0.1 lot, LOSS → lot = 0.1 × 1.3 = 0.13 ✅
Trade 2 (VIRTUAL): 0.13 lot, LOSS → lot = 0.13 × 1.3 = 0.169 ✅
Trade 3 (VIRTUAL): 0.169 lot, LOSS → lot = 0.169 × 1.3 = 0.2197 ✅
Trade 4 (REAL):    0.2197 lot (martingale level 4!) ✅
Trade 5 (REAL):    0.2197 lot, LOSS → lot = 0.2197 × 1.3 = 0.286
```

✅ **Lot size compounds from first loss, arrives at REAL mode already at level 4!**

---

## 📊 Comparison: Old vs New Martingale

### Scenario: 5 consecutive losses, trigger=3

| Trade | Mode (Old) | Lot (Old) | Mode (New) | Lot (New) | Difference |
|-------|-----------|-----------|-----------|-----------|------------|
| 1 LOSS | VIRTUAL | 0.100 | VIRTUAL | 0.100 | Same |
| 2 LOSS | VIRTUAL | 0.100 ❌ | VIRTUAL | 0.130 ✅ | +30% |
| 3 LOSS | VIRTUAL | 0.100 ❌ | VIRTUAL | 0.169 ✅ | +69% |
| 4 LOSS | REAL | 0.100 ❌ | REAL | 0.220 ✅ | +120% |
| 5 LOSS | REAL | 0.130 | REAL | 0.286 ✅ | +120% |

**Key Difference:**
- Old: Trade 4 starts REAL mode at base lot (0.1)
- New: Trade 4 starts REAL mode at martingale level 4 (0.22) ✅

**This is what user wanted!**

---

## 🎯 Mathematical Example

### Setup:
- `base_lot_size`: 0.1
- `martingale_multiplier`: 1.3
- `consecutive_losses_trigger`: 3
- Scenario: 6 consecutive losses, then 1 win

### Old Logic (Buggy):
```
Loss 1: VIRTUAL, 0.100 lot, -$10
Loss 2: VIRTUAL, 0.100 lot, -$10
Loss 3: VIRTUAL, 0.100 lot, -$10
──────────────────────────────────
Trigger = 3, switch to REAL, RESET lot to 0.1 ❌
──────────────────────────────────
Loss 4: REAL, 0.100 lot, -$10
Loss 5: REAL, 0.130 lot, -$13
Loss 6: REAL, 0.169 lot, -$16.9
Win 7:  VIRTUAL, 0.100 lot, +$10

Total: -$10 - $10 - $10 - $10 - $13 - $16.9 + $10 = -$59.9
```

**Result:** Lost $59.9 ❌ (martingale didn't help much)

---

### New Logic (Fixed):
```
Loss 1: VIRTUAL, 0.100 lot, -$10    (lot → 0.130)
Loss 2: VIRTUAL, 0.130 lot, -$13    (lot → 0.169)
Loss 3: VIRTUAL, 0.169 lot, -$16.9  (lot → 0.220)
──────────────────────────────────
Trigger = 3, switch to REAL, KEEP lot = 0.220 ✅
──────────────────────────────────
Loss 4: REAL, 0.220 lot, -$22.0     (lot → 0.286)
Loss 5: REAL, 0.286 lot, -$28.6     (lot → 0.372)
Loss 6: REAL, 0.372 lot, -$37.2     (lot → 0.483)
Win 7:  VIRTUAL, 0.100 lot, +$10    (back to base)

Total: -$10 - $13 - $16.9 - $22.0 - $28.6 - $37.2 + $10 = -$107.7
```

**Wait, this is WORSE?** 🤔

Yes! New logic is MORE AGGRESSIVE with losses. **But the martingale effect kicks in on wins:**

```
Loss 1: VIRTUAL, 0.100 lot, -$10    (lot → 0.130)
Loss 2: VIRTUAL, 0.130 lot, -$13    (lot → 0.169)
Loss 3: VIRTUAL, 0.169 lot, -$16.9  (lot → 0.220)
──────────────────────────────────
Trigger = 3, switch to REAL
──────────────────────────────────
WIN 4:  REAL, 0.220 lot, +$44.0 ✅  (2:1 R:R, recovered!)

Total: -$10 - $13 - $16.9 + $44.0 = +$4.1 profit ✅
```

**With R:R = 2:1, a single win at martingale level 4 recovers all losses + profit!**

---

## 🚨 Important Notes

### 1. Higher Risk Profile

New martingale is MORE AGGRESSIVE:
- Lot sizes grow FASTER
- Drawdowns can be LARGER before recovery
- Max lot size limit is MORE important

**Make sure `max_lot_size` is set appropriately!**

### 2. When It Works Best

New martingale works best when:
- ✅ Strategy has decent win rate (40%+)
- ✅ R:R ratio >= 2:1
- ✅ Losses are clustered (not spread out)
- ✅ Win can come within 1-2 more trades after trigger

**If losses continue for 10+ trades, account can blow up faster!**

### 3. Risk Management

Recommended settings:
```python
'consecutive_losses_trigger': 3,  # Not too low (dangerous)
'martingale_multiplier': 1.3,     # Not too high (exponential!)
'max_lot_size': 10.0,             # Hard cap to prevent blow-up
```

**Lower is safer:**
- `trigger`: 3-5 (NOT 1-2!)
- `multiplier`: 1.2-1.4 (NOT 1.5+!)

---

## ✅ Summary of Fixes

### Fix #1: SL/TP Configuration ✅
- **Problem:** STRATEGY_CONFIG not used, caused confusion
- **Solution:** Marked as LEGACY, clarified to edit BACKTEST_CONFIG
- **How to use:** Edit `atr_sl_multiplier` and `atr_tp_multiplier` in BACKTEST_CONFIG

### Fix #2: Martingale Logic ✅
- **Problem:** Martingale only in REAL mode, lot size reset when switching
- **Solution:** Martingale on EVERY loss, compounds BEFORE switching to REAL
- **Effect:** More aggressive, lot size already at level N when entering REAL mode

### Config Updates ✅
- **Timeframe:** Changed to H1/H1 (user's optimal)
- **ADX Threshold:** Changed to 35 (user's optimal)
- **Initial Balance:** Changed to $1000
- **Commission:** Changed to $7/lot

---

## 🧪 Testing Recommendations

### Test 1: Verify SL/TP Changes Work

```python
# Edit config.py
BACKTEST_CONFIG = {
    ...
    'atr_sl_multiplier': 2.0,  # Changed from 1.5
    'atr_tp_multiplier': 4.0,  # Changed from 3.0
    ...
}

# Run backtest
python examples/run_backtest.py

# Check results - should be DIFFERENT from before!
# Wider SL/TP = fewer trades, potentially higher win rate
```

### Test 2: Verify Martingale Compounding

```python
# Edit config.py
BACKTEST_CONFIG = {
    ...
    'consecutive_losses_trigger': 3,
    'martingale_multiplier': 1.3,
    'base_lot_size': 0.1,
    ...
}

# Run backtest
python examples/run_backtest.py

# Check CSV output for consecutive losses:
# Loss 1: 0.100 lot (VIRTUAL)
# Loss 2: 0.130 lot (VIRTUAL) ✅ Should be 30% higher!
# Loss 3: 0.169 lot (VIRTUAL) ✅ Should be 69% higher!
# Loss 4: 0.220 lot (REAL) ✅ Should be 120% higher!
```

---

## 📝 User Action Items

1. ✅ **Pull latest code:**
   ```bash
   git pull origin claude/read-document-project-011CUQHRGnqfNqLViNGpmb8S
   ```

2. ✅ **Verify config is correct:**
   - Open `config.py`
   - Check BACKTEST_CONFIG has H1/H1, ADX=35, trigger=3

3. ✅ **Test SL/TP changes:**
   - Try different `atr_sl_multiplier` values (1.5, 2.0, 2.5)
   - Verify results CHANGE (different PnL, different trades)

4. ✅ **Test new martingale:**
   - Run backtest with trigger=3
   - Check CSV output for lot size progression
   - Verify lot sizes compound in VIRTUAL mode

5. ✅ **Compare results:**
   - Old logic: Martingale only in REAL
   - New logic: Martingale from first loss
   - Expect: More aggressive, faster recovery OR faster blow-up

---

## ⚠️ Risk Warning

**New martingale is MORE AGGRESSIVE!**

- Lot sizes grow faster
- Drawdowns can be larger
- Recovery is faster IF win comes
- Blow-up is faster IF losses continue

**Test with small balance first!**

**Monitor max_lot_size hits - if frequently hitting max, reduce multiplier or increase trigger!**

---

**End of Document**

Generated with Claude Code
Date: 2025-10-25
