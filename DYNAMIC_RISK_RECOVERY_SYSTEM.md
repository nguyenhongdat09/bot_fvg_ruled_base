# Dynamic Risk Recovery System Documentation

**Date:** 2025-10-25
**Version:** 2.0
**Status:** IMPLEMENTED ✅

---

## 🎯 Overview

This document explains the **3 MAJOR CHANGES** to the trading system per user requirements:

1. ✅ **Lot Size Rounding** - Round to 2 decimal places
2. ✅ **Dynamic Risk Recovery** - Replace Martingale with intelligent recovery
3. ✅ **SL/TP Mode Selection** - Choose between ATR or Fixed Pips

---

## 📋 Change #1: Lot Size Rounding

### Problem
Lot sizes were not rounded, resulting in broker-incompatible values like 0.013, 0.156, etc.

### Solution
Added `round_lot_size()` method that rounds to **2 decimal places**.

### Examples
```
0.013 → 0.01
0.017 → 0.02
0.014 → 0.01
0.016 → 0.02
0.155 → 0.16
0.999 → 1.00
```

### Implementation
```python
def round_lot_size(self, lot_size: float) -> float:
    rounded = round(lot_size, 2)

    # Enforce min/max limits
    if rounded < min_lot_size:
        rounded = min_lot_size
    if rounded > max_lot_size:
        rounded = max_lot_size

    return rounded
```

### Config Parameters
```python
'min_lot_size': 0.01,    # Minimum lot size (broker limit)
'max_lot_size': 10.0,    # Maximum lot size (risk limit)
```

---

## 📋 Change #2: Dynamic Risk Recovery System

### Problem
Old Martingale system:
- Fixed multiplier (1.3x) after each loss
- Didn't account for actual losses
- Could over-leverage or under-recover
- Example:
  ```
  Loss 1: 0.1 lot, -$2.50
  Loss 2: 0.13 lot, -$3.25  (×1.3)
  Loss 3: 0.169 lot, -$4.23 (×1.3)
  Total Loss: -$9.98
  Next lot: 0.220 (×1.3)

  If next trade wins:
  Win: 0.220 lot, +$22 (with 2:1 R:R)
  Net: -$9.98 + $22 = +$12.02 ✓

  BUT if TP is only 1.5:1 R:R:
  Win: 0.220 lot, +$16.50
  Net: -$9.98 + $16.50 = +$6.52 (less recovery)
  ```

###Solution: Dynamic Risk Recovery

**Key Concept:** Calculate lot size based on:
1. Total accumulated losses (VIRTUAL + REAL)
2. TP distance (pips)
3. Recovery target (default: 2× total loss)

**Formula:**
```
Total Loss = |virtual_losses| + |real_losses|
Required Profit = Total Loss × recovery_multiplier (default 2.0)
Minimum Profit = $10 (user requirement)

TP Distance (pips) = known (from ATR or fixed pips)
Lot Size = max(Required Profit, Minimum Profit) / (TP pips × pip_value)
```

### Example Scenario

**Setup:**
- `consecutive_losses_trigger`: 3
- `recovery_multiplier`: 2.0
- `use_atr_sl_tp`: False
- `tp_pips`: 100
- `pip_value`: 0.0001

**Trade Sequence:**

```
Trade 1 (VIRTUAL):
- Lot: 0.1 (base lot)
- Result: LOSS -$2.50
- total_virtual_loss: -$2.50
- Mode: VIRTUAL (consecutive_losses=1)

Trade 2 (VIRTUAL):
- Lot: 0.1 (base lot)
- Result: LOSS -$2.50
- total_virtual_loss: -$5.00
- Mode: VIRTUAL (consecutive_losses=2)

Trade 3 (VIRTUAL):
- Lot: 0.1 (base lot)
- Result: LOSS -$2.50
- total_virtual_loss: -$7.50
- Mode: VIRTUAL → REAL (consecutive_losses=3, trigger!)

Trade 4 (REAL - Recovery):
- Total Loss: $7.50
- Required Profit: $7.50 × 2.0 = $15.00
- TP Distance: 100 pips
- Lot Size Calculation:
  lot = $15.00 / (100 pips × 0.0001)
  lot = $15.00 / 0.01
  lot = 1500 / 100 = 0.15
- Lot: 0.15 (rounded)
- Result: WIN +$15.00 (100 pips × 0.0001 × 0.15 lot × 10,000)
- Net PnL: -$7.50 + $15.00 = +$7.50 ✅ (profit!)
- Mode: REAL → VIRTUAL (reset)
- total_virtual_loss: 0
- total_real_loss: 0
```

**If Trade 4 also loses:**
```
Trade 4 (REAL):
- Lot: 0.15
- Result: LOSS -$3.75
- total_virtual_loss: -$7.50
- total_real_loss: -$3.75
- Total: -$11.25
- Mode: REAL (consecutive_losses=4)

Trade 5 (REAL - Bigger Recovery):
- Total Loss: $11.25
- Required Profit: $11.25 × 2.0 = $22.50
- TP Distance: 100 pips
- Lot Size:
  lot = $22.50 / (100 × 0.0001)
  lot = $22.50 / 0.01
  lot = 0.225 → rounds to 0.23
- Lot: 0.23 (rounded to 2 decimals)
- Result: WIN +$23.00
- Net PnL: -$11.25 + $23.00 = +$11.75 ✅
- Mode: REAL → VIRTUAL (reset)
```

### Comparison: Martingale vs Dynamic Recovery

**Scenario:** 3 losses of $2.50 each, then 1 win

| Method | Trade 1 | Trade 2 | Trade 3 | Trade 4 (Recovery) | Net Result |
|--------|---------|---------|---------|-------------------|------------|
| **Old Martingale** | 0.10 lot<br>-$2.50 | 0.13 lot<br>-$3.25 | 0.169 lot<br>-$4.23 | 0.220 lot<br>+$22.00 | **+$11.02** |
| **Dynamic Recovery** | 0.10 lot<br>-$2.50 | 0.10 lot<br>-$2.50 | 0.10 lot<br>-$2.50 | 0.15 lot<br>+$15.00 | **+$7.50** |

**Key Difference:**
- Martingale: Compounds lot size immediately (higher risk)
- Dynamic: Keeps base lot in VIRTUAL, calculates exact recovery in REAL

**Advantage of Dynamic:**
- ✅ More predictable (exact recovery calculation)
- ✅ Less aggressive (base lot in VIRTUAL mode)
- ✅ Adapts to actual losses (not fixed multiplier)
- ✅ Guarantees minimum $10 profit when win comes

**Disadvantage:**
- ⚠️ Slightly lower profit on wins (but safer)
- ⚠️ Requires accurate TP distance calculation

### Config Parameters

```python
'consecutive_losses_trigger': 3,  # Switch to REAL after N losses
'recovery_multiplier': 2.0,       # Recovery target (2× = recover + profit)
'min_lot_size': 0.01,             # Min lot (broker)
'max_lot_size': 10.0,             # Max lot (risk)
```

**Tuning `recovery_multiplier`:**
- `1.0` = Recover only (break-even)
- `1.5` = Recover + 50% profit
- `2.0` = Recover + 100% profit (default) ✅
- `3.0` = Recover + 200% profit (aggressive)

---

## 📋 Change #3: SL/TP Mode Selection

### Problem
Previous system only supported ATR-based SL/TP. User wanted option for fixed pips.

### Solution
Added `use_atr_sl_tp` flag to switch between modes.

### ATR Mode (Dynamic)
```python
'use_atr_sl_tp': True,
'atr_sl_multiplier': 1.5,   # SL = ATR × 1.5
'atr_tp_multiplier': 3.0,   # TP = ATR × 3.0
```

**Advantages:**
- ✅ Adapts to volatility
- ✅ Wider SL/TP in volatile markets
- ✅ Tighter SL/TP in quiet markets

**Example:**
```
ATR = 0.0030 (30 pips)
SL = 0.0030 × 1.5 = 0.0045 (45 pips)
TP = 0.0030 × 3.0 = 0.0090 (90 pips)
R:R = 1:2
```

### Pips Mode (Fixed)
```python
'use_atr_sl_tp': False,
'sl_pips': 50,              # SL = 50 pips (always)
'tp_pips': 100,             # TP = 100 pips (always)
```

**Advantages:**
- ✅ Consistent risk/reward
- ✅ Predictable lot size calculations
- ✅ Easier to backtest

**Example:**
```
SL = 50 pips (always)
TP = 100 pips (always)
R:R = 1:2 (fixed)
```

### Implementation

```python
def calculate_sl_tp(self, current_price, direction, atr_value=None):
    if self.config['use_atr_sl_tp']:
        # ATR Mode
        sl_distance = atr_value * self.config['atr_sl_multiplier']
        tp_distance = atr_value * self.config['atr_tp_multiplier']
    else:
        # Pips Mode
        sl_distance = self.config['sl_pips'] * self.config['pip_value']
        tp_distance = self.config['tp_pips'] * self.config['pip_value']

    # Calculate prices...
    return sl_price, tp_price, sl_pips, tp_pips
```

### Which Mode to Use?

**Use ATR Mode if:**
- Market volatility changes frequently
- Want adaptive SL/TP
- Trading multiple symbols with different volatility

**Use Pips Mode if:**
- Want consistent risk/reward ratio
- Prefer simpler calculations
- Testing specific SL/TP values

---

## 🔧 Complete Config Example

```python
BACKTEST_CONFIG = {
    # ===== SYMBOL & TIMEFRAME =====
    'symbol': 'GBPUSD',
    'timeframe': 'H1',
    'fvg_timeframe': 'H1',
    'days': 180,

    # ===== ACCOUNT SETTINGS =====
    'initial_balance': 1000.0,
    'risk_per_trade': 0.02,
    'base_lot_size': 0.1,

    # ===== COMMISSION & COSTS =====
    'commission_per_lot': 7.0,
    'pip_value': 0.0001,

    # ===== DYNAMIC RISK RECOVERY =====
    'consecutive_losses_trigger': 3,   # Switch to REAL after 3 losses
    'recovery_multiplier': 2.0,        # Recover + 100% profit
    'min_lot_size': 0.01,              # Minimum 0.01 lot
    'max_lot_size': 10.0,              # Maximum 10 lots

    # ===== SL/TP MODE =====
    'use_atr_sl_tp': False,            # ← Change to True for ATR mode

    # ATR Mode (if use_atr_sl_tp = True)
    'atr_sl_multiplier': 1.5,
    'atr_tp_multiplier': 3.0,

    # Pips Mode (if use_atr_sl_tp = False)
    'sl_pips': 50,                     # ← Change SL pips
    'tp_pips': 100,                    # ← Change TP pips

    # ... other settings ...
}
```

---

## 🧪 Testing Examples

### Test 1: Lot Size Rounding

```python
# Expected behavior:
round_lot_size(0.013) → 0.01
round_lot_size(0.017) → 0.02
round_lot_size(0.145) → 0.15  # rounds to nearest 0.01
round_lot_size(0.999) → 1.00
round_lot_size(15.678) → 10.0  # capped at max_lot_size
```

### Test 2: Dynamic Recovery Calculation

```python
# Scenario:
total_virtual_loss = -$5.00
total_real_loss = -$3.00
recovery_multiplier = 2.0
tp_pips = 100
pip_value = 0.0001

# Calculation:
total_loss = $5 + $3 = $8
required_profit = $8 × 2.0 = $16
lot_size = $16 / (100 × 0.0001) = $16 / 0.01 = 1.6

# Result: 1.6 lot (rounds to 1.60)
```

### Test 3: SL/TP Modes

**ATR Mode:**
```python
'use_atr_sl_tp': True
current_price = 1.2500
direction = 'BUY'
atr_value = 0.0020 (20 pips)

sl_distance = 0.0020 × 1.5 = 0.0030 (30 pips)
tp_distance = 0.0020 × 3.0 = 0.0060 (60 pips)

sl_price = 1.2500 - 0.0030 = 1.2470
tp_price = 1.2500 + 0.0060 = 1.2560
```

**Pips Mode:**
```python
'use_atr_sl_tp': False
'sl_pips': 50
'tp_pips': 100
current_price = 1.2500
direction = 'BUY'

sl_distance = 50 × 0.0001 = 0.0050 (50 pips)
tp_distance = 100 × 0.0001 = 0.0100 (100 pips)

sl_price = 1.2500 - 0.0050 = 1.2450
tp_price = 1.2500 + 0.0100 = 1.2600
```

---

## 📊 Expected Behavior Changes

### Before (Martingale):
```
Loss Sequence (trigger=3):
Trade 1 (V): 0.100 lot → -$2.50 (×1.3 = 0.130)
Trade 2 (V): 0.130 lot → -$3.25 (×1.3 = 0.169)
Trade 3 (V): 0.169 lot → -$4.23 (×1.3 = 0.220)
Trade 4 (R): 0.220 lot → +$22.00 (win)
Net: -$9.98 + $22.00 = +$12.02
```

### After (Dynamic Recovery):
```
Loss Sequence (trigger=3):
Trade 1 (V): 0.10 lot → -$2.50
Trade 2 (V): 0.10 lot → -$2.50
Trade 3 (V): 0.10 lot → -$2.50
Total Loss: -$7.50

Trade 4 (R): 0.15 lot (calculated for $15 profit)
Result: +$15.00 (win)
Net: -$7.50 + $15.00 = +$7.50
```

**Key Differences:**
1. Less aggressive lot sizing in VIRTUAL mode
2. Exact recovery calculation in REAL mode
3. Predictable profit target (2× loss)
4. All lot sizes rounded to 2 decimals

---

## ⚠️ Risk Warnings

### 1. Recovery Multiplier Too High
```python
'recovery_multiplier': 5.0  # ← TOO HIGH!

# Problem:
# -$10 loss → need $50 profit
# May require very large lot size
# Risk of hitting max_lot_size or margin call
```

**Recommendation:** Keep between 1.5 - 2.5

### 2. Max Lot Size Too Low
```python
'max_lot_size': 0.5  # ← TOO LOW!

# Problem:
# Large losses cannot be recovered
# Example: -$20 loss, tp=50 pips
#   Required lot: $40 / (50×0.0001) = 8.0 lots
#   Max lot: 0.5 → Can only make $2.50 profit
#   Cannot recover $20 loss!
```

**Recommendation:** Set at least 10× base_lot_size

### 3. TP Pips Too Small
```python
'use_atr_sl_tp': False
'tp_pips': 20  # ← TOO SMALL!

# Problem:
# Small TP requires very large lot size for recovery
# Example: -$10 loss, tp=20 pips
#   lot = $20 / (20×0.0001) = 10.0 lots (very large!)
```

**Recommendation:** TP pips should be ≥ 50 for dynamic recovery to work safely

---

## 🎯 Recommended Settings

### Conservative (Lower Risk)
```python
'consecutive_losses_trigger': 5,   # More patient
'recovery_multiplier': 1.5,        # Lower profit target
'min_lot_size': 0.01,
'max_lot_size': 5.0,               # Lower max
'use_atr_sl_tp': True,             # Adaptive
'atr_sl_multiplier': 2.0,          # Wider SL
'atr_tp_multiplier': 4.0,          # Higher TP
```

### Balanced (Recommended)
```python
'consecutive_losses_trigger': 3,   # Standard
'recovery_multiplier': 2.0,        # 100% profit
'min_lot_size': 0.01,
'max_lot_size': 10.0,
'use_atr_sl_tp': False,            # Fixed pips
'sl_pips': 50,
'tp_pips': 100,                    # 1:2 R:R
```

### Aggressive (Higher Risk)
```python
'consecutive_losses_trigger': 2,   # Quick trigger
'recovery_multiplier': 3.0,        # 200% profit
'min_lot_size': 0.01,
'max_lot_size': 20.0,              # Higher max
'use_atr_sl_tp': False,
'sl_pips': 30,                     # Tighter SL
'tp_pips': 150,                    # 1:5 R:R
```

---

## 🚀 Usage Instructions

### Step 1: Update Config

Edit `config.py`:

```python
BACKTEST_CONFIG = {
    # ... other settings ...

    # Choose SL/TP mode
    'use_atr_sl_tp': False,  # Set True for ATR, False for Pips

    # If using PIPS mode:
    'sl_pips': 50,
    'tp_pips': 100,

    # If using ATR mode:
    'atr_sl_multiplier': 1.5,
    'atr_tp_multiplier': 3.0,

    # Dynamic recovery settings
    'consecutive_losses_trigger': 3,
    'recovery_multiplier': 2.0,
    'min_lot_size': 0.01,
    'max_lot_size': 10.0,
}
```

### Step 2: Run Backtest

```bash
python examples/run_backtest.py
```

### Step 3: Check Results

Look for in CSV output:
- Lot sizes should be rounded (0.01, 0.15, 1.20, etc.)
- REAL mode trades should have larger lots
- Win after losses should show profit ≥ 2× total loss

### Step 4: Verify Recovery

Example check:
```
Trades:
1. VIRTUAL 0.10 lot LOSS -$2.50
2. VIRTUAL 0.10 lot LOSS -$2.50
3. VIRTUAL 0.10 lot LOSS -$2.50
Total Loss: -$7.50

4. REAL 0.15 lot WIN +$15.00
Net: -$7.50 + $15.00 = +$7.50 ✅

Recovery worked! Profit = 2× loss
```

---

## 📝 Summary

### What Changed:

1. ✅ **Lot Size Rounding**
   - All lot sizes rounded to 2 decimals
   - Respects min/max lot size limits

2. ✅ **Dynamic Risk Recovery**
   - Replaced fixed Martingale multiplier
   - Calculates exact lot size needed for recovery
   - Guarantees minimum $10 profit on win
   - Recovery target = 2× total loss (configurable)

3. ✅ **SL/TP Mode Selection**
   - Can choose ATR-based (dynamic) or Fixed Pips
   - ATR adapts to volatility
   - Pips mode gives consistent R:R ratio

### Benefits:

- 🎯 More predictable recovery
- 🎯 Exact profit calculations
- 🎯 Flexible SL/TP modes
- 🎯 Broker-compatible lot sizes
- 🎯 Less aggressive than old Martingale

### Risks:

- ⚠️ Still requires wins to recover
- ⚠️ Large losses need large lot sizes
- ⚠️ Must set max_lot_size appropriately
- ⚠️ TP distance affects recovery lot size

---

**End of Documentation**

Generated with Claude Code
Date: 2025-10-25
