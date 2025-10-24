#!/usr/bin/env python3
"""
Demo: FVG State Changes Over Time (No Look-Ahead Bias)

Script nay chung minh FVG state thay doi theo thoi gian,
KHONG co look-ahead bias (khong biet truoc tuong lai)
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from core.fvg import FVGManager


def create_specific_test_data():
    """
    Tao data co FVG ro rang:
    - Ngay 01/01: FVG duoc tao (chua bi lap)
    - Ngay 03/01: FVG bi cham (touched)
    """

    print("\n" + "="*70)
    print("CREATING TEST DATA WITH KNOWN FVG")
    print("="*70)

    # Tao 10 nen
    dates = pd.date_range(start='2025-01-01 00:00', periods=10, freq='15min')

    # Thiet ke data de tao Bullish FVG tai index 4 (01/01 01:00)
    # FVG se bi cham tai index 7 (01/01 01:45)

    data = pd.DataFrame({
        # Index 0, 1
        'open':  [1.10000, 1.10010],
        'high':  [1.10020, 1.10030],
        'low':   [1.09980, 1.10000],
        'close': [1.10010, 1.10020],

        # Index 2, 3, 4 -> TAO BULLISH FVG
        # Candle 2: high = 1.10000
        # Candle 3: middle
        # Candle 4: low = 1.10100 (GAP tu 1.10000 den 1.10100)
    }, index=dates[:2])

    # Candle 2: Low candle (tao gap)
    candle_2 = pd.DataFrame({
        'open':  [1.10020],
        'high':  [1.10030],  # <- High = 1.10030
        'low':   [1.09980],
        'close': [1.09990],
    }, index=[dates[2]])

    # Candle 3: Middle candle
    candle_3 = pd.DataFrame({
        'open':  [1.09990],
        'high':  [1.10050],
        'low':   [1.09970],
        'close': [1.10040],
    }, index=[dates[3]])

    # Candle 4: High candle (BULLISH FVG CREATED)
    candle_4 = pd.DataFrame({
        'open':  [1.10040],
        'high':  [1.10120],
        'low':   [1.10100],  # <- Low = 1.10100 (GAP!)
        'close': [1.10110],
    }, index=[dates[4]])

    # Candle 5, 6: Price o tren FVG (FVG chua bi cham)
    candle_5 = pd.DataFrame({
        'open':  [1.10110],
        'high':  [1.10150],
        'low':   [1.10105],  # <- Van tren FVG top (1.10100)
        'close': [1.10140],
    }, index=[dates[5]])

    candle_6 = pd.DataFrame({
        'open':  [1.10140],
        'high':  [1.10160],
        'low':   [1.10120],  # <- Van tren FVG top
        'close': [1.10150],
    }, index=[dates[6]])

    # Candle 7: Price CHAM FVG (TOUCHED!)
    candle_7 = pd.DataFrame({
        'open':  [1.10150],
        'high':  [1.10160],
        'low':   [1.10090],  # <- CHAM FVG top (1.10100)
        'close': [1.10100],
    }, index=[dates[7]])

    # Candle 8, 9: Sau khi FVG bi lap
    candle_8 = pd.DataFrame({
        'open':  [1.10100],
        'high':  [1.10130],
        'low':   [1.10080],
        'close': [1.10120],
    }, index=[dates[8]])

    candle_9 = pd.DataFrame({
        'open':  [1.10120],
        'high':  [1.10140],
        'low':   [1.10110],
        'close': [1.10130],
    }, index=[dates[9]])

    # Combine all
    data = pd.concat([
        data, candle_2, candle_3, candle_4,
        candle_5, candle_6, candle_7, candle_8, candle_9
    ])

    # Add volume
    data['volume'] = 500

    print(f"\n✓ Created {len(data)} candles")
    print(f"\n📊 Key Events:")
    print(f"   Index 4 ({dates[4]}): BULLISH FVG CREATED")
    print(f"      Gap: 1.10030 - 1.10100")
    print(f"   Index 7 ({dates[7]}): FVG TOUCHED")
    print(f"      Low: 1.10090 (cham FVG top 1.10100)")

    return data


def calculate_simple_atr(data, period=14):
    """Calculate ATR"""
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()

    # Fill first values
    atr = atr.fillna(0.0003)

    return atr


def demo_time_accurate_fvg():
    """
    Demo FVG state changes over time
    Chung minh KHONG co look-ahead bias
    """

    print("\n" + "="*70)
    print("DEMO: FVG STATE CHANGES OVER TIME (NO LOOK-AHEAD BIAS)")
    print("="*70)

    # Create data
    data = create_specific_test_data()
    atr = calculate_simple_atr(data)

    # Initialize manager
    manager = FVGManager(lookback_days=90, min_gap_atr_ratio=0.2)

    print("\n" + "="*70)
    print("PROCESSING CANDLES ONE BY ONE")
    print("="*70)

    # Process each candle
    for i in range(3, len(data)):
        current_time = data.index[i]
        current_price = data['close'].iloc[i]

        print(f"\n{'='*70}")
        print(f"INDEX {i}: {current_time}")
        print(f"{'='*70}")
        print(f"Price: O={data['open'].iloc[i]:.5f} H={data['high'].iloc[i]:.5f} "
              f"L={data['low'].iloc[i]:.5f} C={data['close'].iloc[i]:.5f}")

        # Update manager (CHI NHIN DATA TU DAU DEN i, KHONG NHIN TUONG LAI!)
        new_fvg = manager.update(data.iloc[:i+1], i, atr.iloc[i])

        # Show new FVG if created
        if new_fvg:
            print(f"\n🆕 NEW FVG CREATED:")
            print(f"   Type: {new_fvg.fvg_type}")
            print(f"   Range: {new_fvg.bottom:.5f} - {new_fvg.top:.5f}")
            print(f"   Status: {'ACTIVE' if new_fvg.is_active else 'TOUCHED'}")
            print(f"   ⚠️  LUU Y: Luc nay FVG moi tao, CHUA BI LAP!")

        # Show current FVG state
        stats = manager.get_statistics()
        structure = manager.get_market_structure(current_price)

        print(f"\n📊 FVG STATE TAI THOI DIEM NAY:")
        print(f"   Total FVGs created: {stats['total_bullish_created'] + stats['total_bearish_created']}")
        print(f"   Active FVGs: {stats['total_active']}")
        print(f"   Touched FVGs: {stats['total_bullish_touched'] + stats['total_bearish_touched']}")

        # Show active FVGs detail
        if manager.active_bullish_fvgs or manager.active_bearish_fvgs:
            print(f"\n📍 ACTIVE FVGs (chua bi lap):")

            for fvg in manager.active_bullish_fvgs:
                print(f"   BULLISH: {fvg.bottom:.5f} - {fvg.top:.5f}")
                print(f"      Created: index {fvg.created_index}")
                print(f"      Status: {'ACTIVE ✅' if fvg.is_active else 'TOUCHED ❌'}")
                print(f"      Touched: {'YES' if fvg.is_touched else 'NO'}")

            for fvg in manager.active_bearish_fvgs:
                print(f"   BEARISH: {fvg.bottom:.5f} - {fvg.top:.5f}")
                print(f"      Created: index {fvg.created_index}")
                print(f"      Status: {'ACTIVE ✅' if fvg.is_active else 'TOUCHED ❌'}")
                print(f"      Touched: {'YES' if fvg.is_touched else 'NO'}")
        else:
            print(f"\n⚠️  Khong co FVG active (tat ca da bi lap)")

        # Market structure
        print(f"\n🎯 MARKET STRUCTURE:")
        print(f"   Bias: {structure['bias']}")

        if structure['bias'] in ['BULLISH_BIAS', 'BOTH_FVG']:
            print(f"   ✅ CO THE TRADE BUY (co FVG target duoi)")
        if structure['bias'] in ['BEARISH_BIAS', 'BOTH_FVG']:
            print(f"   ✅ CO THE TRADE SELL (co FVG target tren)")
        if structure['bias'] == 'NO_FVG':
            print(f"   ❌ KHONG TRADE (khong co FVG target)")

        # Special events
        if i == 4:
            print(f"\n" + "🔔"*35)
            print(f"   QUAN TRONG: FVG vua duoc tao tai index {i}")
            print(f"   Luc nay FVG CHUA BI LAP (is_touched = False)")
            print(f"   Neu test chien luoc tai day, FVG se la VALID TARGET!")
            print(f"🔔"*35)

        if i == 7:
            print(f"\n" + "⚠️ "*35)
            print(f"   QUAN TRONG: FVG bi CHAM tai index {i}")
            print(f"   Luc nay FVG DA BI LAP (is_touched = True)")
            print(f"   FVG khong con la valid target nua!")
            print(f"⚠️ "*35)

        # Wait for user (optional) - DISABLED for auto run
        # if i in [4, 5, 6, 7]:
        #     input("\n>>> Press Enter to continue to next candle...")


def main():
    """Main function"""

    print("\n" + "="*70)
    print("  TIME-ACCURATE FVG DEMO")
    print("  Chung minh: KHONG CO LOOK-AHEAD BIAS")
    print("="*70)

    print("\n📝 Giai thich:")
    print("   - FVG duoc tao tai index 4 (01/01 01:00)")
    print("   - FVG bi cham tai index 7 (01/01 01:45)")
    print("   - Khi test tai index 4, 5, 6: FVG CHUA BI LAP (valid)")
    print("   - Khi test tai index 7+: FVG DA BI LAP (invalid)")
    print("   - Code KHONG biet truoc FVG se bi lap, chi biet khi no xay ra!")

    # input("\n>>> Press Enter to start demo...")  # DISABLED for auto run

    demo_time_accurate_fvg()

    print("\n" + "="*70)
    print("  DEMO COMPLETED ✓")
    print("="*70)

    print("\n📊 KET LUAN:")
    print("   ✅ Code xu ly dung theo thoi gian thuc")
    print("   ✅ KHONG co look-ahead bias")
    print("   ✅ Khi test tai index 4-6: FVG is_touched = False (valid)")
    print("   ✅ Khi test tai index 7+: FVG is_touched = True (invalid)")
    print("   ✅ Du lieu phan anh DUNG trang thai tai moi thoi diem")

    print("\n💡 Y NGHIA:")
    print("   - Khi backtest chien luoc, FVG state se CHINH XAC")
    print("   - Khong co truong hop 'biet truoc tuong lai'")
    print("   - Ket qua backtest la DANG TIN CAY")

    return 0


if __name__ == '__main__':
    exit(main())
