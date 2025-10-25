#!/usr/bin/env python3
"""
Quick Start Guide - Modifying SL/TP Configuration

This script shows the simplest way to modify SL/TP settings
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BACKTEST_CONFIG


def demo_modify_config():
    """Demonstrate how to modify configuration"""
    
    print("\n" + "="*60)
    print("QUICK START: MODIFYING SL/TP CONFIGURATION")
    print("="*60)
    
    print("\n📋 Current Configuration:")
    print("-" * 60)
    print(f"Mode: {BACKTEST_CONFIG['sl_tp_mode']}")
    print(f"Fixed SL: {BACKTEST_CONFIG['fixed_sl_pips']} pips")
    print(f"Fixed TP: {BACKTEST_CONFIG['fixed_tp_pips']} pips")
    print(f"ATR SL: {BACKTEST_CONFIG['atr_sl_multiplier']} × ATR")
    print(f"ATR TP: {BACKTEST_CONFIG['atr_tp_multiplier']} × ATR")
    
    print("\n\n" + "="*60)
    print("METHOD 1: Edit config.py directly")
    print("="*60)
    
    print("""
Open config.py and modify BACKTEST_CONFIG:

# For Fixed Mode:
BACKTEST_CONFIG = {
    'sl_tp_mode': 'fixed',  # ← Change this
    'fixed_sl_pips': 15.0,  # ← Change this (was 20.0)
    'fixed_tp_pips': 30.0,  # ← Change this (was 40.0)
    ...
}

# For ATR Mode:
BACKTEST_CONFIG = {
    'sl_tp_mode': 'atr',    # ← Change this
    'atr_sl_multiplier': 2.0,  # ← Change this (was 1.5)
    'atr_tp_multiplier': 4.0,  # ← Change this (was 3.0)
    ...
}
""")
    
    print("\n" + "="*60)
    print("METHOD 2: Create custom config in your script")
    print("="*60)
    
    print("""
from config import BACKTEST_CONFIG
from examples.run_backtest import run_backtest

# Create a copy and modify
my_config = BACKTEST_CONFIG.copy()

# Example 1: Tight stops for scalping
my_config['sl_tp_mode'] = 'fixed'
my_config['fixed_sl_pips'] = 10.0
my_config['fixed_tp_pips'] = 20.0

# Run backtest
backtester, metrics = run_backtest(my_config, mode='fixed')

# Example 2: Wide stops for swing trading
my_config['fixed_sl_pips'] = 50.0
my_config['fixed_tp_pips'] = 100.0
backtester, metrics = run_backtest(my_config, mode='fixed')

# Example 3: ATR-based adaptive stops
my_config['sl_tp_mode'] = 'atr'
my_config['atr_sl_multiplier'] = 2.0
my_config['atr_tp_multiplier'] = 4.0
backtester, metrics = run_backtest(my_config, mode='atr')
""")
    
    print("\n" + "="*60)
    print("COMMON CONFIGURATIONS")
    print("="*60)
    
    configs = [
        ("Scalping", "fixed", 10, 20, "-", "-"),
        ("Day Trading", "fixed", 20, 40, "-", "-"),
        ("Swing Trading", "fixed", 50, 100, "-", "-"),
        ("Conservative ATR", "atr", "-", "-", 1.0, 2.0),
        ("Standard ATR", "atr", "-", "-", 1.5, 3.0),
        ("Aggressive ATR", "atr", "-", "-", 2.0, 4.0),
    ]
    
    print(f"\n{'Strategy':<20} {'Mode':<8} {'SL Pips':<10} {'TP Pips':<10} {'SL ATR':<10} {'TP ATR':<10}")
    print("-" * 75)
    for name, mode, sl_pips, tp_pips, sl_atr, tp_atr in configs:
        print(f"{name:<20} {mode:<8} {sl_pips!s:<10} {tp_pips!s:<10} {sl_atr!s:<10} {tp_atr!s:<10}")
    
    print("\n\n" + "="*60)
    print("PRACTICAL EXAMPLES")
    print("="*60)
    
    print("""
# Example: Test if tighter stops improve results
configs_to_test = [
    {'sl': 10, 'tp': 20},
    {'sl': 15, 'tp': 30},
    {'sl': 20, 'tp': 40},
    {'sl': 25, 'tp': 50},
]

for cfg in configs_to_test:
    my_config = BACKTEST_CONFIG.copy()
    my_config['fixed_sl_pips'] = cfg['sl']
    my_config['fixed_tp_pips'] = cfg['tp']
    
    backtester, metrics = run_backtest(my_config, mode='fixed')
    print(f"SL={cfg['sl']}, TP={cfg['tp']}: Win Rate={metrics['win_rate']:.2f}%")
""")
    
    print("\n" + "="*60)
    print("TIPS FOR OPTIMIZATION")
    print("="*60)
    
    print("""
1. Risk:Reward Ratio
   - TP should be 2-3x SL for good profit factor
   - Example: SL=20, TP=40 (1:2 ratio)
   - Example: SL=20, TP=60 (1:3 ratio)

2. Fixed vs ATR
   - Fixed: Better for stable markets
   - ATR: Better for volatile markets
   - ATR adapts to market conditions automatically

3. Testing Strategy
   - Start with standard settings (20/40 pips or 1.5x/3.0x ATR)
   - Test variations ±25% (e.g., 15/30, 20/40, 25/50)
   - Compare win rate and profit factor
   - Choose configuration with best risk-adjusted returns

4. Market Conditions
   - High volatility → Use ATR or wider fixed stops
   - Low volatility → Use fixed or tighter stops
   - Trending → Wider TP (e.g., 1:3 ratio)
   - Ranging → Tighter TP (e.g., 1:2 ratio)
""")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("""
1. Choose your preferred method (edit config.py or create custom config)
2. Run: python examples/run_backtest.py
3. Check results and adjust
4. Test multiple configurations: python examples/test_config_variations.py
5. Analyze trade logs in logs/backtest_trades_*.csv

For detailed documentation, see: examples/README.md
""")


if __name__ == '__main__':
    demo_modify_config()
