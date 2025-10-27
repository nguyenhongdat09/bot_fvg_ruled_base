#!/usr/bin/env python3
"""
Test FVG Modules with Real MT5 Data

Script test FVG voi du lieu that tu MetaTrader 5
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_data_filepath, FVG_CONFIG, CHARTS_DIR
from core.fvg import FVGManager, FVGVisualizer


def calculate_atr(data, period=14):
    """Calculate ATR"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def load_real_data(filepath=None):
    """
    Load real data from CSV

    Args:
        filepath: Path to CSV file (optional, use default if None)

    Returns:
        pd.DataFrame: Data or None if not found
    """
    print("\n" + "="*60)
    print("LOADING REAL MT5 DATA")
    print("="*60)

    if filepath is None:
        filepath = get_data_filepath()

    print(f"\n📁 Looking for: {filepath}")

    if not filepath.exists():
        print(f"❌ File not found!")
        print(f"\n💡 Please download data first:")
        print(f"   python data/download_mt5_data.py")
        return None

    print(f"✓ File found")

    # Load CSV
    print(f"\n📊 Loading data...")
    data = pd.read_csv(filepath, index_col='time', parse_dates=True)

    print(f"✓ Data loaded")
    print(f"   Rows: {len(data)}")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Duration: {(data.index[-1] - data.index[0]).days} days")

    return data


def test_fvg_with_real_data(data):
    """
    Test FVG modules with real data

    Args:
        data: Real market data DataFrame

    Returns:
        FVGManager: Manager with processed FVGs
    """
    print("\n" + "="*60)
    print("TESTING FVG MODULES WITH REAL DATA")
    print("="*60)

    # 1. Calculate ATR
    print("\n1. Calculating ATR...")
    atr = calculate_atr(data, period=14)
    print(f"✓ ATR calculated")
    print(f"   Mean ATR: {atr.mean():.5f}")
    print(f"   Min ATR: {atr.min():.5f}")
    print(f"   Max ATR: {atr.max():.5f}")

    # 2. Initialize FVG Manager
    print("\n2. Initializing FVG Manager...")
    manager = FVGManager(
        lookback_days=FVG_CONFIG['lookback_days'],
        min_gap_atr_ratio=FVG_CONFIG['min_gap_atr_ratio'],
        min_gap_pips=FVG_CONFIG['min_gap_pips']
    )
    print(f"✓ Manager initialized")
    print(f"   Lookback: {FVG_CONFIG['lookback_days']} days")
    print(f"   Min gap ATR ratio: {FVG_CONFIG['min_gap_atr_ratio']}")

    # 3. Process all candles
    print("\n3. Processing candles...")
    print(f"   Total candles: {len(data)}")

    # Show progress every 10%
    progress_step = max(1, len(data) // 10)

    for i in range(20, len(data)):
        manager.update(data.iloc[:i+1], i, atr.iloc[i])

        if i % progress_step == 0:
            progress = (i / len(data)) * 100
            stats = manager.get_statistics()
            print(f"   [{progress:.0f}%] Candle {i}/{len(data)}: {stats['total_active']} active FVGs")

    print("✓ Processing completed")

    # 4. Final statistics
    print("\n4. FVG Statistics:")
    stats = manager.get_statistics()

    print(f"\n   Created:")
    print(f"      Bullish: {stats['total_bullish_created']}")
    print(f"      Bearish: {stats['total_bearish_created']}")
    print(f"      Total: {stats['total_bullish_created'] + stats['total_bearish_created']}")

    print(f"\n   Status:")
    print(f"      Active: {stats['total_active']}")
    print(f"      Touched: {stats['total_bullish_touched'] + stats['total_bearish_touched']}")

    print(f"\n   Touch Rate:")
    print(f"      Bullish: {stats['bullish_touch_rate']:.1f}%")
    print(f"      Bearish: {stats['bearish_touch_rate']:.1f}%")
    print(f"      Overall: {(stats['bullish_touch_rate'] + stats['bearish_touch_rate']) / 2:.1f}%")

    # 5. Current market structure
    print("\n5. Current Market Structure:")
    current_price = data['close'].iloc[-1]
    structure = manager.get_market_structure(current_price)

    print(f"\n   Current Price: {current_price:.5f}")
    print(f"   Market Bias: {structure['bias']}")
    print(f"   Active FVGs: {structure['total_active_fvgs']}")
    print(f"   Bullish FVGs below: {len(structure['bullish_fvgs_below'])}")
    print(f"   Bearish FVGs above: {len(structure['bearish_fvgs_above'])}")

    if structure['nearest_bullish_target']:
        fvg = structure['nearest_bullish_target']
        distance = fvg.get_distance_to_price(current_price)
        print(f"\n   📍 Nearest Bullish Target:")
        print(f"      Range: {fvg.bottom:.5f} - {fvg.top:.5f}")
        print(f"      Distance: {distance:.5f} ({distance * 10000:.1f} pips)")
        print(f"      Created: {fvg.created_timestamp}")
        print(f"      Age: {fvg.get_age_in_days(data.index[-1]):.1f} days")

    if structure['nearest_bearish_target']:
        fvg = structure['nearest_bearish_target']
        distance = fvg.get_distance_to_price(current_price)
        print(f"\n   📍 Nearest Bearish Target:")
        print(f"      Range: {fvg.bottom:.5f} - {fvg.top:.5f}")
        print(f"      Distance: {distance:.5f} ({distance * 10000:.1f} pips)")
        print(f"      Created: {fvg.created_timestamp}")
        print(f"      Age: {fvg.get_age_in_days(data.index[-1]):.1f} days")

    # 6. Trading signals based on FVG
    print("\n6. Trading Signals (based on FVG only):")
    from core.fvg import validate_signal_with_fvg, get_fvg_target

    for signal in ['BUY', 'SELL']:
        is_valid = validate_signal_with_fvg(structure, signal)
        target = get_fvg_target(structure, signal)

        if is_valid and target:
            print(f"\n   ✓ {signal} Signal:")
            print(f"      Valid: {is_valid}")
            print(f"      Target FVG: {target.bottom:.5f} - {target.top:.5f}")
            print(f"      Potential: {abs(target.middle - current_price) * 10000:.1f} pips")
        else:
            print(f"\n   ✗ {signal} Signal: Not valid (no FVG target)")

    return manager


def create_visualizations(data, manager):
    """
    Create visualizations for real data

    Args:
        data: Market data
        manager: FVG Manager with processed FVGs
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Create visualizer
    visualizer = FVGVisualizer(
        show_touched_fvgs=FVG_CONFIG['show_touched_fvgs'],
        show_labels=FVG_CONFIG['show_labels']
    )

    # Create output directory
    output_dir = CHARTS_DIR / 'real_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📊 Creating charts...")
    print(f"   Output: {output_dir}")

    # Get FVGs
    all_fvgs = manager.all_fvgs_history

    # Create full report
    files = visualizer.create_fvg_report(
        data=data,
        fvgs=all_fvgs,
        signals=None,  # No signals yet
        output_dir=str(output_dir)
    )

    print(f"\n✓ Charts created:")
    for name, path in files.items():
        file_size = os.path.getsize(path) / 1024
        print(f"   {name}: {path} ({file_size:.1f} KB)")

    return files


def export_data(manager):
    """
    Export FVG data to CSV

    Args:
        manager: FVG Manager
    """
    print("\n" + "="*60)
    print("EXPORTING DATA")
    print("="*60)

    # Export history
    history_df = manager.export_history_to_dataframe()
    if not history_df.empty:
        csv_path = Path('logs/fvg_real_data_history.csv')
        history_df.to_csv(csv_path, index=False)
        print(f"\n✓ FVG history: {csv_path}")
        print(f"   Records: {len(history_df)}")

    # Export active
    active_df = manager.export_active_to_dataframe()
    if not active_df.empty:
        csv_path = Path('logs/fvg_real_data_active.csv')
        active_df.to_csv(csv_path, index=False)
        print(f"\n✓ Active FVGs: {csv_path}")
        print(f"   Records: {len(active_df)}")
    else:
        print(f"\n⚠️  No active FVGs to export")


def main():
    """Main function"""

    print("\n" + "="*70)
    print("  FVG MODULES - REAL DATA TEST")
    print("="*70)

    # 1. Load real data
    data = load_real_data()
    if data is None:
        return 1

    # 2. Test FVG with real data
    manager = test_fvg_with_real_data(data)

    # 3. Create visualizations
    chart_files = create_visualizations(data, manager)

    # 4. Export data
    export_data(manager)

    # 5. Summary
    print("\n" + "="*70)
    print("  TEST COMPLETED ✓")
    print("="*70)

    print(f"\n📊 Summary:")
    stats = manager.get_statistics()
    print(f"   Total FVGs created: {stats['total_bullish_created'] + stats['total_bearish_created']}")
    print(f"   Active FVGs: {stats['total_active']}")
    print(f"   Touch rate: {(stats['bullish_touch_rate'] + stats['bearish_touch_rate']) / 2:.1f}%")

    print(f"\n📁 Output files:")
    print(f"   Charts: {CHARTS_DIR / 'real_data'}")
    print(f"   CSV: logs/fvg_real_data_*.csv")

    print(f"\n📝 Next steps:")
    print(f"   1. Open charts in browser to verify FVG detection")
    print(f"   2. Review CSV files for data validation")
    print(f"   3. Analyze FVG patterns in real market")
    print(f"   4. Integrate with indicators for signal generation")

    return 0


if __name__ == '__main__':
    exit(main())
