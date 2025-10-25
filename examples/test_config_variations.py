#!/usr/bin/env python3
"""
Test different SL/TP configurations

Demonstrates how to modify BACKTEST_CONFIG for different scenarios
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BACKTEST_CONFIG
from examples.run_backtest import run_backtest, create_sample_data


def test_fixed_mode_variations():
    """Test different fixed pip configurations"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT FIXED SL/TP CONFIGURATIONS")
    print("="*60)
    
    configs = [
        {'fixed_sl_pips': 10.0, 'fixed_tp_pips': 20.0, 'name': 'Tight (10/20 pips)'},
        {'fixed_sl_pips': 20.0, 'fixed_tp_pips': 40.0, 'name': 'Standard (20/40 pips)'},
        {'fixed_sl_pips': 30.0, 'fixed_tp_pips': 60.0, 'name': 'Wide (30/60 pips)'},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {cfg['name']}")
        print(f"{'='*60}")
        
        # Create a modified config
        test_config = BACKTEST_CONFIG.copy()
        test_config['sl_tp_mode'] = 'fixed'
        test_config['fixed_sl_pips'] = cfg['fixed_sl_pips']
        test_config['fixed_tp_pips'] = cfg['fixed_tp_pips']
        
        print(f"SL: {cfg['fixed_sl_pips']} pips, TP: {cfg['fixed_tp_pips']} pips")
        
        # Run backtest
        backtester, metrics = run_backtest(test_config, mode='fixed')
        
        results.append({
            'name': cfg['name'],
            'sl_pips': cfg['fixed_sl_pips'],
            'tp_pips': cfg['fixed_tp_pips'],
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'return': metrics['total_return_pct'],
        })
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON - FIXED MODE VARIATIONS")
    print(f"{'='*60}\n")
    
    print(f"{'Configuration':<20} {'Trades':<10} {'Win Rate':<12} {'Return %':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['trades']:<10} {r['win_rate']:<12.2f} {r['return']:<10.2f}")


def test_atr_mode_variations():
    """Test different ATR multiplier configurations"""
    print("\n\n" + "="*60)
    print("TESTING DIFFERENT ATR MULTIPLIER CONFIGURATIONS")
    print("="*60)
    
    configs = [
        {'sl_mult': 1.0, 'tp_mult': 2.0, 'name': 'Conservative (1x/2x)'},
        {'sl_mult': 1.5, 'tp_mult': 3.0, 'name': 'Standard (1.5x/3x)'},
        {'sl_mult': 2.0, 'tp_mult': 4.0, 'name': 'Aggressive (2x/4x)'},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {cfg['name']}")
        print(f"{'='*60}")
        
        # Create a modified config
        test_config = BACKTEST_CONFIG.copy()
        test_config['sl_tp_mode'] = 'atr'
        test_config['atr_sl_multiplier'] = cfg['sl_mult']
        test_config['atr_tp_multiplier'] = cfg['tp_mult']
        
        print(f"SL: ATR × {cfg['sl_mult']}, TP: ATR × {cfg['tp_mult']}")
        
        # Run backtest
        backtester, metrics = run_backtest(test_config, mode='atr')
        
        results.append({
            'name': cfg['name'],
            'sl_mult': cfg['sl_mult'],
            'tp_mult': cfg['tp_mult'],
            'trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'return': metrics['total_return_pct'],
        })
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON - ATR MODE VARIATIONS")
    print(f"{'='*60}\n")
    
    print(f"{'Configuration':<22} {'Trades':<10} {'Win Rate':<12} {'Return %':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<22} {r['trades']:<10} {r['win_rate']:<12.2f} {r['return']:<10.2f}")


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("SL/TP CONFIGURATION TESTING")
    print("="*60)
    print("\nThis script demonstrates how to modify SL/TP settings")
    print("in BACKTEST_CONFIG and test different configurations.\n")
    
    # Test fixed mode variations
    test_fixed_mode_variations()
    
    # Test ATR mode variations
    test_atr_mode_variations()
    
    print("\n\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    
    print("\n💡 Key Takeaways:")
    print("  1. Fixed mode: Adjust 'fixed_sl_pips' and 'fixed_tp_pips' in config.py")
    print("  2. ATR mode: Adjust 'atr_sl_multiplier' and 'atr_tp_multiplier' in config.py")
    print("  3. Toggle mode: Set 'sl_tp_mode' to 'fixed' or 'atr' in config.py")
    print("  4. Each configuration produces different trading results")
    print("\n📝 To modify settings:")
    print("  - Edit config.py BACKTEST_CONFIG section")
    print("  - Or create custom config dict and pass to run_backtest()")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
