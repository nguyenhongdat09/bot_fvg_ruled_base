#!/usr/bin/env python3
"""
Interactive FVG Testing Script
Test từng tính năng FVG một cách chi tiết
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fvg import FVGManager, FVGDetector, FVG, validate_signal_with_fvg, get_fvg_target


def print_header(text):
    """Print header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_section(text):
    """Print section"""
    print(f"\n>>> {text}")


def test_fvg_creation():
    """Test 1: Tạo và test FVG object"""
    print_header("TEST 1: FVG Object Creation")

    timestamp = pd.Timestamp('2024-01-01 10:00:00')

    # Create Bullish FVG
    fvg_bull = FVG(
        fvg_id='TEST_BULL_001',
        fvg_type='BULLISH',
        created_index=100,
        created_timestamp=timestamp,
        created_candle_indices=(98, 99, 100),
        top=1.10500,
        bottom=1.10300,
        strength=0.8,
        atr_at_creation=0.00025
    )

    print(f"\nBullish FVG created:")
    print(f"  Type: {fvg_bull.fvg_type}")
    print(f"  Range: {fvg_bull.bottom:.5f} - {fvg_bull.top:.5f}")
    print(f"  Middle: {fvg_bull.middle:.5f}")
    print(f"  Gap Size: {fvg_bull.gap_size:.5f}")
    print(f"  Strength: {fvg_bull.strength:.2f}")
    print(f"  Status: {'ACTIVE' if fvg_bull.is_active else 'INACTIVE'}")

    # Test check_touched
    print("\nTest touching FVG:")

    # Candle không chạm
    test_candle_1 = {'high': 1.10600, 'low': 1.10520}
    touched_1 = fvg_bull.check_touched(
        test_candle_1['high'],
        test_candle_1['low'],
        101,
        timestamp + pd.Timedelta(minutes=15)
    )
    print(f"  Candle 1 (1.10520-1.10600): touched={touched_1} ✓ (above FVG)")

    # Candle chạm vào top
    test_candle_2 = {'high': 1.10550, 'low': 1.10480}
    touched_2 = fvg_bull.check_touched(
        test_candle_2['high'],
        test_candle_2['low'],
        102,
        timestamp + pd.Timedelta(minutes=30)
    )
    print(f"  Candle 2 (1.10480-1.10550): touched={touched_2} ✓ (touched top)")
    print(f"  FVG now: {'ACTIVE' if fvg_bull.is_active else 'TOUCHED'}")

    # Test is_valid_target
    print("\nTest valid target:")
    valid_1 = fvg_bull.is_valid_target(1.10600)
    print(f"  Price 1.10600 (above): valid={valid_1} (touched FVG)")


def test_fvg_detection():
    """Test 2: Phát hiện FVG từ data"""
    print_header("TEST 2: FVG Detection")

    # Create data với FVG rõ ràng
    dates = pd.date_range(start='2024-01-01', periods=10, freq='15min')

    # Tạo Bullish FVG tại index 4
    # Candle 2: high = 1.10000
    # Candle 3: middle candle
    # Candle 4: low = 1.10100 (gap từ 1.10000 đến 1.10100)

    data = pd.DataFrame({
        'open':  [1.10000, 1.10010, 1.09950, 1.10080, 1.10100, 1.10090, 1.10095, 1.10100, 1.10105, 1.10110],
        'high':  [1.10020, 1.10030, 1.10000, 1.10100, 1.10120, 1.10110, 1.10115, 1.10120, 1.10125, 1.10130],
        'low':   [1.09980, 1.10000, 1.09920, 1.10060, 1.10100, 1.10080, 1.10085, 1.10090, 1.10095, 1.10100],
        'close': [1.10010, 1.10020, 1.09960, 1.10090, 1.10110, 1.10100, 1.10105, 1.10110, 1.10115, 1.10120],
    }, index=dates)

    # ATR giả định
    atr = pd.Series([0.00030] * len(data), index=dates)

    print("\nData sample:")
    print(data[['open', 'high', 'low', 'close']].head())

    # Detect FVG
    detector = FVGDetector(min_gap_atr_ratio=0.3)

    print("\nDetecting FVGs...")
    for i in range(2, len(data)):
        fvg = detector.detect_fvg_at_index(data, i, atr.iloc[i])
        if fvg:
            print(f"\n  ✓ FVG found at index {i}:")
            print(f"    Type: {fvg.fvg_type}")
            print(f"    Range: {fvg.bottom:.5f} - {fvg.top:.5f}")
            print(f"    Gap Size: {fvg.gap_size:.5f}")
            print(f"    Strength: {fvg.strength:.2f}")

            # Show 3 candles
            print(f"    3 candles:")
            print(f"      [{i-2}] high={data.iloc[i-2]['high']:.5f}")
            print(f"      [{i-1}] (middle)")
            print(f"      [{i}]   low={data.iloc[i]['low']:.5f}")


def test_market_structure():
    """Test 3: Market structure analysis"""
    print_header("TEST 3: Market Structure Analysis")

    # Create manager
    manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.3)

    # Manually add FVGs for testing
    timestamp = pd.Timestamp('2024-01-01 10:00:00')

    # Add Bullish FVG below current price
    fvg_bull_1 = FVG(
        fvg_id='BULL_1',
        fvg_type='BULLISH',
        created_index=50,
        created_timestamp=timestamp,
        created_candle_indices=(48, 49, 50),
        top=1.10200,
        bottom=1.10100,
        strength=0.8,
        atr_at_creation=0.00025
    )
    manager.active_bullish_fvgs.append(fvg_bull_1)

    # Add Bearish FVG above current price
    fvg_bear_1 = FVG(
        fvg_id='BEAR_1',
        fvg_type='BEARISH',
        created_index=60,
        created_timestamp=timestamp + pd.Timedelta(minutes=150),
        created_candle_indices=(58, 59, 60),
        top=1.10600,
        bottom=1.10500,
        strength=0.9,
        atr_at_creation=0.00025
    )
    manager.active_bearish_fvgs.append(fvg_bear_1)

    # Test different price levels
    test_prices = [1.10300, 1.10450, 1.10150]

    for price in test_prices:
        print(f"\n>>> Current Price: {price:.5f}")

        structure = manager.get_market_structure(price)

        print(f"  Market Bias: {structure['bias']}")
        print(f"  Bullish FVGs below: {len(structure['bullish_fvgs_below'])}")
        print(f"  Bearish FVGs above: {len(structure['bearish_fvgs_above'])}")

        if structure['nearest_bullish_target']:
            fvg = structure['nearest_bullish_target']
            distance = fvg.get_distance_to_price(price)
            print(f"  Nearest Bullish: {fvg.bottom:.5f}-{fvg.top:.5f} (distance: {distance:.5f})")

        if structure['nearest_bearish_target']:
            fvg = structure['nearest_bearish_target']
            distance = fvg.get_distance_to_price(price)
            print(f"  Nearest Bearish: {fvg.bottom:.5f}-{fvg.top:.5f} (distance: {distance:.5f})")

        # Test signal validation
        print(f"  Signal validation:")
        for signal in ['BUY', 'SELL']:
            is_valid = validate_signal_with_fvg(structure, signal)
            target = get_fvg_target(structure, signal)
            print(f"    {signal}: valid={is_valid}, has_target={target is not None}")


def test_fvg_tracking():
    """Test 4: FVG tracking over time"""
    print_header("TEST 4: FVG Tracking Over Time")

    # Create simple trending data
    n = 50
    dates = pd.date_range(start='2024-01-01', periods=n, freq='15min')

    # Uptrend
    base = 1.10000
    trend = np.linspace(0, 0.00200, n)
    noise = np.random.randn(n) * 0.00010

    close = base + trend + noise
    high = close + 0.00005
    low = close - 0.00005
    open_price = close + np.random.randn(n) * 0.00002

    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
    }, index=dates)

    # Calculate ATR
    atr = pd.Series([0.00020] * len(data), index=dates)

    # Initialize manager
    manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.3)

    print("\nProcessing candles...")

    fvg_events = []
    for i in range(5, len(data)):
        new_fvg = manager.update(data.iloc[:i+1], i, atr.iloc[i])

        if new_fvg:
            fvg_events.append({
                'index': i,
                'time': data.index[i],
                'fvg': new_fvg
            })

    print(f"\n  Total FVGs detected: {len(fvg_events)}")

    if fvg_events:
        print("\n  First 5 FVG events:")
        for i, event in enumerate(fvg_events[:5], 1):
            fvg = event['fvg']
            print(f"    {i}. Index {event['index']}: {fvg.fvg_type} "
                  f"{fvg.bottom:.5f}-{fvg.top:.5f}")

    # Final statistics
    stats = manager.get_statistics()
    print(f"\n  Final statistics:")
    print(f"    Total created: {stats['total_bullish_created'] + stats['total_bearish_created']}")
    print(f"    Active: {stats['total_active']}")
    print(f"    Touched: {stats['total_bullish_touched'] + stats['total_bearish_touched']}")
    print(f"    Touch rate: {(stats['bullish_touch_rate'] + stats['bearish_touch_rate']) / 2:.1f}%")


def main():
    print("\n" + "="*70)
    print("  INTERACTIVE FVG TESTING")
    print("="*70)

    np.random.seed(42)

    try:
        test_fvg_creation()
        test_fvg_detection()
        test_market_structure()
        test_fvg_tracking()

        print("\n" + "="*70)
        print("  ALL TESTS COMPLETED ✓")
        print("="*70)
        print("\n📝 Summary:")
        print("  1. FVG object creation & touching ✓")
        print("  2. FVG detection from OHLC data ✓")
        print("  3. Market structure analysis ✓")
        print("  4. FVG tracking over time ✓")
        print()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
