#!/usr/bin/env python3
"""
Simple FVG Test Script
Test nhanh các chức năng FVG cơ bản
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fvg import FVGManager, FVGDetector, FVG


def create_test_data(n=100):
    """Tạo data test đơn giản"""
    dates = pd.date_range(start='2024-01-01', periods=n, freq='15min')

    # Tạo giá với trend
    base = 1.10000
    trend = np.linspace(0, 0.001, n)
    noise = np.random.randn(n) * 0.0003

    close = base + trend + noise
    high = close + np.abs(np.random.randn(n)) * 0.0002
    low = close - np.abs(np.random.randn(n)) * 0.0002
    open_price = close + np.random.randn(n) * 0.0001

    return pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)


def calculate_simple_atr(data, period=14):
    """Tính ATR đơn giản"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def main():
    print("\n" + "="*60)
    print("FVG SIMPLE TEST")
    print("="*60)

    # 1. Tạo data
    print("\n1. Tạo test data...")
    data = create_test_data(n=100)
    print(f"   ✓ {len(data)} candles")
    print(f"   Price range: {data['low'].min():.5f} - {data['high'].max():.5f}")

    # 2. Tính ATR
    print("\n2. Tính ATR...")
    atr = calculate_simple_atr(data)
    print(f"   ✓ ATR mean: {atr.mean():.5f}")

    # 3. Test FVG Detector
    print("\n3. Test FVG Detector...")
    detector = FVGDetector(min_gap_atr_ratio=0.3)
    fvgs = detector.detect_all_fvgs(data, atr, start_index=15)

    print(f"   ✓ Detected {len(fvgs)} FVGs")
    stats = detector.get_statistics(fvgs)
    print(f"     - Bullish: {stats['bullish']}")
    print(f"     - Bearish: {stats['bearish']}")
    print(f"     - Avg Strength: {stats['avg_strength']:.2f}")

    # 4. Test FVG Manager
    print("\n4. Test FVG Manager...")
    manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.3)

    # Process all candles
    for i in range(15, len(data)):
        manager.update(data.iloc[:i+1], i, atr.iloc[i])

    mgr_stats = manager.get_statistics()
    print(f"   ✓ Manager statistics:")
    print(f"     - Total created: {mgr_stats['total_bullish_created'] + mgr_stats['total_bearish_created']}")
    print(f"     - Active: {mgr_stats['total_active']}")
    print(f"     - Touched: {mgr_stats['total_bullish_touched'] + mgr_stats['total_bearish_touched']}")

    # 5. Test Market Structure
    print("\n5. Test Market Structure...")
    current_price = data['close'].iloc[-1]
    structure = manager.get_market_structure(current_price)

    print(f"   Current price: {current_price:.5f}")
    print(f"   Market Bias: {structure['bias']}")
    print(f"   Bullish FVGs below: {len(structure['bullish_fvgs_below'])}")
    print(f"   Bearish FVGs above: {len(structure['bearish_fvgs_above'])}")

    if structure['nearest_bullish_target']:
        fvg = structure['nearest_bullish_target']
        print(f"   Nearest Bullish Target: {fvg.bottom:.5f} - {fvg.top:.5f}")

    if structure['nearest_bearish_target']:
        fvg = structure['nearest_bearish_target']
        print(f"   Nearest Bearish Target: {fvg.bottom:.5f} - {fvg.top:.5f}")

    # 6. Show some FVGs
    print("\n6. Sample FVGs:")
    all_fvgs = manager.get_all_active_fvgs()

    if all_fvgs:
        for i, fvg in enumerate(all_fvgs[:3], 1):
            print(f"   {i}. {fvg.fvg_type}: {fvg.bottom:.5f} - {fvg.top:.5f}")
            print(f"      Created: {fvg.created_timestamp}")
            print(f"      Strength: {fvg.strength:.2f}")
            print(f"      Distance: {fvg.get_distance_to_price(current_price):.5f}")
    else:
        print("   No active FVGs")

    # 7. Export data
    print("\n7. Export test data...")
    history_df = manager.export_history_to_dataframe()

    if not history_df.empty:
        csv_path = 'logs/fvg_simple_test.csv'
        os.makedirs('logs', exist_ok=True)
        history_df.to_csv(csv_path, index=False)
        print(f"   ✓ Saved to: {csv_path}")
        print(f"   Records: {len(history_df)}")

    print("\n" + "="*60)
    print("TEST COMPLETED ✓")
    print("="*60)

    return 0


if __name__ == '__main__':
    np.random.seed(42)
    exit(main())
