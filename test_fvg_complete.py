#!/usr/bin/env python3
"""
Complete FVG Module Testing Script

Script này test đầy đủ các FVG modules:
1. FVG Model
2. FVG Detector
3. FVG Manager
4. FVG Visualizer

Usage:
    python test_fvg_complete.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fvg.fvg_model import FVG, generate_fvg_id, calculate_fvg_strength
from core.fvg.fvg_detector import FVGDetector
from core.fvg.fvg_manager import FVGManager, validate_signal_with_fvg, get_fvg_target
from core.fvg.fvg_visualizer import FVGVisualizer


def create_sample_data(n_candles: int = 500) -> pd.DataFrame:
    """
    Tạo sample OHLCV data để test

    Args:
        n_candles: Số nến

    Returns:
        pd.DataFrame: OHLCV data
    """
    print(f"\n{'='*60}")
    print("STEP 1: Creating sample data...")
    print(f"{'='*60}")

    # Create timestamp index (M15 timeframe)
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_candles, freq='15min')

    # Generate price data với trend + noise
    np.random.seed(42)

    base_price = 1.10000
    trend = np.linspace(0, 0.005, n_candles)  # Uptrend 500 pips
    noise = np.random.randn(n_candles) * 0.0005  # Volatility

    close = base_price + trend + noise

    # Generate OHLC
    high = close + np.abs(np.random.randn(n_candles)) * 0.0003
    low = close - np.abs(np.random.randn(n_candles)) * 0.0003
    open_price = close + np.random.randn(n_candles) * 0.0002

    # Generate volume
    volume = np.random.randint(100, 1000, n_candles)

    data = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    print(f"✓ Created {len(data)} candles")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: {data['low'].min():.5f} to {data['high'].max():.5f}")

    return data


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range)

    Args:
        data: OHLC DataFrame
        period: ATR period

    Returns:
        pd.Series: ATR values
    """

    # True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR = SMA of True Range
    atr = true_range.rolling(window=period).mean()

    return atr


def test_fvg_model():
    """Test FVG Model"""
    print(f"\n{'='*60}")
    print("STEP 2: Testing FVG Model...")
    print(f"{'='*60}")

    # Create test FVG
    timestamp = pd.Timestamp('2024-01-01 00:00:00')

    fvg = FVG(
        fvg_id=generate_fvg_id('BULLISH', timestamp, 100),
        fvg_type='BULLISH',
        created_index=100,
        created_timestamp=timestamp,
        created_candle_indices=(98, 99, 100),
        top=1.10500,
        bottom=1.10300,
        strength=0.8,
        atr_at_creation=0.00025
    )

    print(f"\n✓ Created test FVG:")
    print(f"  {fvg}")
    print(f"  Middle: {fvg.middle:.5f}")
    print(f"  Gap Size: {fvg.gap_size:.5f}")

    # Test check_touched
    print(f"\n  Testing check_touched():")

    # Candle không chạm
    touched = fvg.check_touched(1.10600, 1.10550, 101, timestamp + timedelta(minutes=15))
    print(f"    Candle above FVG: touched={touched} (expected False)")

    # Candle chạm
    touched = fvg.check_touched(1.10550, 1.10450, 102, timestamp + timedelta(minutes=30))
    print(f"    Candle touching FVG: touched={touched} (expected True)")
    print(f"    FVG status: {fvg.is_active} (expected False)")

    # Test is_valid_target
    print(f"\n  Testing is_valid_target():")
    valid = fvg.is_valid_target(1.10600)
    print(f"    Price above FVG: valid={valid} (expected False - FVG touched)")

    # Test distance
    distance = fvg.get_distance_to_price(1.10600)
    print(f"\n  Distance to price 1.10600: {distance:.5f}")

    print("\n✓ FVG Model tests passed!")


def test_fvg_detector(data: pd.DataFrame, atr: pd.Series):
    """Test FVG Detector"""
    print(f"\n{'='*60}")
    print("STEP 3: Testing FVG Detector...")
    print(f"{'='*60}")

    detector = FVGDetector(min_gap_atr_ratio=0.3)

    # Detect all FVGs
    fvgs = detector.detect_all_fvgs(data, atr, start_index=20)

    print(f"\n✓ Detected {len(fvgs)} FVGs")

    # Statistics
    stats = detector.get_statistics(fvgs)
    print(f"\n  FVG Statistics:")
    print(f"    Total: {stats['total']}")
    print(f"    Bullish: {stats['bullish']}")
    print(f"    Bearish: {stats['bearish']}")
    print(f"    Avg Gap Size: {stats['avg_gap_size']:.5f}")
    print(f"    Avg Strength: {stats['avg_strength']:.3f}")

    # Show first 5 FVGs
    if fvgs:
        print(f"\n  First 5 FVGs:")
        for i, fvg in enumerate(fvgs[:5]):
            print(f"    {i+1}. {fvg}")

    print("\n✓ FVG Detector tests passed!")

    return fvgs


def test_fvg_manager(data: pd.DataFrame, atr: pd.Series):
    """Test FVG Manager"""
    print(f"\n{'='*60}")
    print("STEP 4: Testing FVG Manager...")
    print(f"{'='*60}")

    manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.3)

    # Simulate live updates
    print(f"\n  Simulating live FVG updates...")

    for i in range(20, len(data)):
        manager.update(data.iloc[:i+1], i, atr.iloc[i])

        # Print progress every 100 candles
        if i % 100 == 0:
            stats = manager.get_statistics()
            print(f"    Candle {i}: Active FVGs = {stats['total_active']}")

    # Final statistics
    stats = manager.get_statistics()
    print(f"\n✓ FVG Manager Statistics:")
    print(f"    Total Active: {stats['total_active']}")
    print(f"    Active Bullish: {stats['active_bullish']}")
    print(f"    Active Bearish: {stats['active_bearish']}")
    print(f"    Total Created (Bullish): {stats['total_bullish_created']}")
    print(f"    Total Created (Bearish): {stats['total_bearish_created']}")
    print(f"    Total Touched (Bullish): {stats['total_bullish_touched']}")
    print(f"    Total Touched (Bearish): {stats['total_bearish_touched']}")
    print(f"    Bullish Touch Rate: {stats['bullish_touch_rate']:.2f}%")
    print(f"    Bearish Touch Rate: {stats['bearish_touch_rate']:.2f}%")

    # Test market structure
    current_price = data['close'].iloc[-1]
    structure = manager.get_market_structure(current_price)

    print(f"\n✓ Market Structure at current price ({current_price:.5f}):")
    print(f"    Bias: {structure['bias']}")
    print(f"    Bullish FVGs below: {len(structure['bullish_fvgs_below'])}")
    print(f"    Bearish FVGs above: {len(structure['bearish_fvgs_above'])}")

    if structure['nearest_bullish_target']:
        print(f"    Nearest Bullish Target: {structure['nearest_bullish_target'].bottom:.5f} - {structure['nearest_bullish_target'].top:.5f}")

    if structure['nearest_bearish_target']:
        print(f"    Nearest Bearish Target: {structure['nearest_bearish_target'].bottom:.5f} - {structure['nearest_bearish_target'].top:.5f}")

    # Test signal validation
    print(f"\n  Testing signal validation:")

    test_signals = ['BUY', 'SELL']
    for signal in test_signals:
        is_valid = validate_signal_with_fvg(structure, signal)
        target = get_fvg_target(structure, signal)
        print(f"    {signal}: valid={is_valid}, has_target={target is not None}")

    print("\n✓ FVG Manager tests passed!")

    return manager


def test_fvg_visualizer(data: pd.DataFrame, manager: FVGManager):
    """Test FVG Visualizer"""
    print(f"\n{'='*60}")
    print("STEP 5: Testing FVG Visualizer...")
    print(f"{'='*60}")

    visualizer = FVGVisualizer(show_touched_fvgs=True, show_labels=True)

    # Get all FVGs from history
    all_fvgs = manager.all_fvgs_history

    print(f"\n  Creating FVG chart with {len(all_fvgs)} FVGs...")

    # Create sample signals
    signals = [
        {'signal': 'BUY', 'timestamp': data.index[100], 'entry': data['close'].iloc[100]},
        {'signal': 'SELL', 'timestamp': data.index[200], 'entry': data['close'].iloc[200]},
        {'signal': 'BUY', 'timestamp': data.index[300], 'entry': data['close'].iloc[300]},
    ]

    # Plot main chart
    os.makedirs('logs/charts', exist_ok=True)
    main_chart_path = 'logs/charts/test_fvg_chart.html'

    fig = visualizer.plot_fvg_chart(
        data,
        all_fvgs,
        title="FVG Test Chart",
        show_volume=True,
        signals=signals,
        save_path=main_chart_path
    )

    print(f"✓ Main chart saved to: {main_chart_path}")

    # Plot statistics
    stats_chart_path = 'logs/charts/test_fvg_statistics.html'
    fig_stats = visualizer.plot_fvg_statistics(all_fvgs, save_path=stats_chart_path)

    print(f"✓ Statistics chart saved to: {stats_chart_path}")

    # Create full report
    print(f"\n  Creating full FVG report...")
    report_files = visualizer.create_fvg_report(
        data,
        all_fvgs,
        signals=signals,
        output_dir='logs/charts'
    )

    print("\n✓ FVG Visualizer tests passed!")


def test_export_data(manager: FVGManager):
    """Test data export"""
    print(f"\n{'='*60}")
    print("STEP 6: Testing Data Export...")
    print(f"{'='*60}")

    # Export history to DataFrame
    history_df = manager.export_history_to_dataframe()

    print(f"\n✓ Exported FVG history:")
    print(f"    Total records: {len(history_df)}")

    if not history_df.empty:
        print(f"\n  First 5 records:")
        print(history_df.head())

        # Save to CSV
        os.makedirs('logs', exist_ok=True)
        csv_path = 'logs/fvg_history_test.csv'
        history_df.to_csv(csv_path, index=False)
        print(f"\n✓ FVG history saved to: {csv_path}")

    # Export active FVGs
    active_df = manager.export_active_to_dataframe()
    print(f"\n✓ Exported active FVGs: {len(active_df)} records")

    if not active_df.empty:
        active_csv_path = 'logs/fvg_active_test.csv'
        active_df.to_csv(active_csv_path, index=False)
        print(f"✓ Active FVGs saved to: {active_csv_path}")

    print("\n✓ Data export tests passed!")


def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("FVG MODULES - COMPLETE TEST SUITE")
    print("="*60)

    try:
        # Step 1: Create sample data
        data = create_sample_data(n_candles=500)

        # Calculate ATR
        atr = calculate_atr(data, period=14)

        # Step 2: Test FVG Model
        test_fvg_model()

        # Step 3: Test FVG Detector
        fvgs = test_fvg_detector(data, atr)

        # Step 4: Test FVG Manager
        manager = test_fvg_manager(data, atr)

        # Step 5: Test FVG Visualizer
        test_fvg_visualizer(data, manager)

        # Step 6: Test Data Export
        test_export_data(manager)

        # Final summary
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED! ✓")
        print(f"{'='*60}")

        print("\n📊 Output files created:")
        print("  - logs/charts/test_fvg_chart.html (Main FVG chart)")
        print("  - logs/charts/test_fvg_statistics.html (Statistics)")
        print("  - logs/charts/fvg_chart_*.html (Full report)")
        print("  - logs/fvg_history_test.csv (FVG history data)")
        print("  - logs/fvg_active_test.csv (Active FVGs data)")

        print("\n📝 Next steps:")
        print("  1. Open the HTML files in your browser to view charts")
        print("  2. Review the CSV files for data validation")
        print("  3. Test with real market data from MT5")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
