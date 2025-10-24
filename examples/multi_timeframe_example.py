"""
MULTI-TIMEFRAME FVG STRATEGY EXAMPLE

Vi du nay minh hoa 2 cach ket hop FVG voi cac indicator tren nhieu timeframe:
1. Option 1: Manual Resample & Align (don gian)
2. Option 2: MultiTimeframeManager (chuyen nghiep)

Strategy:
- FVG: Phan tich tren H1
- RSI: Phan tich tren M15
- Signal: BUY khi H1 FVG bullish + M15 RSI < 30

Author: Claude Code
Date: 2025-10-24
"""

import sys
import os

# Add parent directory to path to import core modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import pandas_ta as ta
from core.fvg.fvg_manager import FVGManager
from core.fvg.multi_timeframe_manager import MultiTimeframeManager
from config import FVG_CONFIG


def option1_manual_resample_align():
    """
    Option 1: Manual Resample & Align

    Uu diem: Don gian, de hieu, de debug
    Nhuoc diem: Code dai, khong reusable
    """

    print("=" * 80)
    print("OPTION 1: MANUAL RESAMPLE & ALIGN")
    print("=" * 80)

    # ===== STEP 1: Load M15 data (base timeframe) =====
    print("\nStep 1: Loading M15 data...")
    m15_data = pd.read_csv('data/EURUSD_M15_30days.csv', index_col=0, parse_dates=True)
    print(f"  Loaded {len(m15_data)} M15 candles")
    print(f"  Date range: {m15_data.index[0]} to {m15_data.index[-1]}")

    # ===== STEP 2: Resample to H1 =====
    print("\nStep 2: Resampling to H1...")
    h1_data = m15_data.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    print(f"  Resampled to {len(h1_data)} H1 candles")

    # ===== STEP 3: Calculate indicators =====
    print("\nStep 3: Calculating indicators...")

    # H1 ATR for FVG
    h1_data['atr'] = ta.atr(h1_data['high'], h1_data['low'], h1_data['close'], length=14)

    # M15 RSI
    m15_data['rsi'] = ta.rsi(m15_data['close'], length=14)

    print(f"  H1 ATR calculated")
    print(f"  M15 RSI calculated")

    # ===== STEP 4: Process H1 FVG sequentially =====
    print("\nStep 4: Processing H1 FVG...")
    fvg_h1 = FVGManager(
        lookback_days=FVG_CONFIG['lookback_days'],
        min_gap_atr_ratio=FVG_CONFIG['min_gap_atr_ratio']
    )

    h1_states = {}  # {timestamp: structure}

    for i in range(20, len(h1_data)):
        # Sequential processing - no look-ahead bias
        fvg_h1.update(h1_data.iloc[:i+1], i, h1_data.iloc[i]['atr'])
        structure = fvg_h1.get_market_structure(h1_data.iloc[i]['close'])

        h1_states[h1_data.index[i]] = {
            'bias': structure['bias'],
            'total_active_fvgs': structure['total_active_fvgs'],
            'nearest_bullish': structure['nearest_bullish_target'],
            'nearest_bearish': structure['nearest_bearish_target']
        }

    print(f"  Processed {len(h1_states)} H1 candles")
    print(f"  H1 FVG Statistics:")
    stats = fvg_h1.get_statistics()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # ===== STEP 5: Align H1 states to M15 (forward fill) =====
    print("\nStep 5: Aligning H1 states to M15...")
    m15_data['h1_bias'] = None

    for i in range(len(m15_data)):
        m15_time = m15_data.index[i]

        # Find latest H1 state BEFORE or AT m15_time (forward fill)
        valid_h1_times = [t for t in h1_states.keys() if t <= m15_time]

        if valid_h1_times:
            latest_h1_time = max(valid_h1_times)
            m15_data.loc[m15_time, 'h1_bias'] = h1_states[latest_h1_time]['bias']

    print(f"  Aligned H1 states to {len(m15_data)} M15 candles")

    # ===== STEP 6: Backtest strategy on M15 =====
    print("\nStep 6: Backtesting strategy...")
    signals = []

    for i in range(100, len(m15_data)):
        # Get aligned data at M15 index i
        h1_bias = m15_data.iloc[i]['h1_bias']
        m15_rsi = m15_data.iloc[i]['rsi']
        m15_price = m15_data.iloc[i]['close']
        m15_time = m15_data.index[i]

        # Skip if H1 data not available yet
        if pd.isna(h1_bias):
            continue

        # Strategy: BUY when H1 bullish + M15 RSI oversold
        if h1_bias == 'BULLISH_BIAS' and m15_rsi < 30:
            signals.append({
                'time': m15_time,
                'type': 'BUY',
                'price': m15_price,
                'h1_bias': h1_bias,
                'm15_rsi': m15_rsi
            })

        # Strategy: SELL when H1 bearish + M15 RSI overbought
        elif h1_bias == 'BEARISH_BIAS' and m15_rsi > 70:
            signals.append({
                'time': m15_time,
                'type': 'SELL',
                'price': m15_price,
                'h1_bias': h1_bias,
                'm15_rsi': m15_rsi
            })

    # ===== STEP 7: Results =====
    print("\n" + "=" * 80)
    print("RESULTS - OPTION 1")
    print("=" * 80)
    print(f"Total signals: {len(signals)}")
    print(f"  BUY signals: {len([s for s in signals if s['type'] == 'BUY'])}")
    print(f"  SELL signals: {len([s for s in signals if s['type'] == 'SELL'])}")

    if signals:
        print(f"\nFirst 5 signals:")
        for sig in signals[:5]:
            print(f"  {sig['type']:4s} at {sig['time']} | "
                  f"Price: {sig['price']:.5f} | "
                  f"H1 Bias: {sig['h1_bias']:15s} | "
                  f"M15 RSI: {sig['m15_rsi']:.2f}")

    return signals


def option2_multi_timeframe_manager():
    """
    Option 2: MultiTimeframeManager

    Uu diem: Clean, reusable, chuyen nghiep
    Nhuoc diem: Phuc tap hon khi implement (nhung da co san roi!)
    """

    print("\n\n")
    print("=" * 80)
    print("OPTION 2: MULTI-TIMEFRAME MANAGER")
    print("=" * 80)

    # ===== STEP 1: Load M15 data =====
    print("\nStep 1: Loading M15 data...")
    m15_data = pd.read_csv('data/EURUSD_M15_30days.csv', index_col=0, parse_dates=True)
    print(f"  Loaded {len(m15_data)} M15 candles")

    # ===== STEP 2: Initialize MultiTimeframeManager =====
    print("\nStep 2: Initializing MultiTimeframeManager...")
    mtf = MultiTimeframeManager(m15_data, base_timeframe='M15')

    # ===== STEP 3: Add FVG timeframes =====
    print("\nStep 3: Adding FVG timeframes...")
    mtf.add_fvg_timeframe('H1', lookback_days=FVG_CONFIG['lookback_days'])
    mtf.add_fvg_timeframe('H4', lookback_days=FVG_CONFIG['lookback_days'])

    # ===== STEP 4: Calculate M15 indicators =====
    print("\nStep 4: Calculating M15 indicators...")
    m15_data['rsi'] = ta.rsi(m15_data['close'], length=14)
    m15_data['volume_sma'] = ta.sma(m15_data['volume'], length=20)

    # ===== STEP 5: Backtest strategy =====
    print("\nStep 5: Backtesting strategy...")
    signals = []

    for i in range(100, len(m15_data)):
        # Update all timeframes up to index i
        mtf.update(i)

        # Get FVG bias from multiple timeframes
        h1_bias = mtf.get_fvg_bias('H1', i)
        h4_bias = mtf.get_fvg_bias('H4', i)

        # Get M15 indicators
        m15_rsi = m15_data.iloc[i]['rsi']
        m15_volume = m15_data.iloc[i]['volume']
        m15_volume_sma = m15_data.iloc[i]['volume_sma']
        m15_price = m15_data.iloc[i]['close']
        m15_time = m15_data.index[i]

        # Skip if not enough data
        if h1_bias is None or h4_bias is None or pd.isna(m15_rsi):
            continue

        # Strategy: BUY when H1+H4 bullish + M15 RSI oversold + Volume spike
        volume_spike = m15_volume > m15_volume_sma * 1.5

        if (h1_bias == 'BULLISH_BIAS' and
            h4_bias == 'BULLISH_BIAS' and
            m15_rsi < 30 and
            volume_spike):

            signals.append({
                'time': m15_time,
                'type': 'BUY',
                'price': m15_price,
                'h1_bias': h1_bias,
                'h4_bias': h4_bias,
                'm15_rsi': m15_rsi,
                'volume_ratio': m15_volume / m15_volume_sma
            })

        # Strategy: SELL when H1+H4 bearish + M15 RSI overbought + Volume spike
        elif (h1_bias == 'BEARISH_BIAS' and
              h4_bias == 'BEARISH_BIAS' and
              m15_rsi > 70 and
              volume_spike):

            signals.append({
                'time': m15_time,
                'type': 'SELL',
                'price': m15_price,
                'h1_bias': h1_bias,
                'h4_bias': h4_bias,
                'm15_rsi': m15_rsi,
                'volume_ratio': m15_volume / m15_volume_sma
            })

    # ===== STEP 6: Results =====
    print("\n" + "=" * 80)
    print("RESULTS - OPTION 2")
    print("=" * 80)
    print(f"Total signals: {len(signals)}")
    print(f"  BUY signals: {len([s for s in signals if s['type'] == 'BUY'])}")
    print(f"  SELL signals: {len([s for s in signals if s['type'] == 'SELL'])}")

    if signals:
        print(f"\nFirst 5 signals:")
        for sig in signals[:5]:
            print(f"  {sig['type']:4s} at {sig['time']} | "
                  f"Price: {sig['price']:.5f} | "
                  f"H1: {sig['h1_bias']:15s} | "
                  f"H4: {sig['h4_bias']:15s} | "
                  f"RSI: {sig['m15_rsi']:.2f} | "
                  f"Vol: {sig['volume_ratio']:.2f}x")

    # ===== STEP 7: Statistics =====
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    for tf in mtf.get_available_timeframes():
        print(f"\n{tf} FVG Statistics:")
        stats = mtf.get_statistics(tf)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    return signals, mtf


def verify_no_look_ahead_bias():
    """
    Verify: Multi-timeframe strategy khong bi look-ahead bias

    Test: Ket qua backtest khong thay doi khi them data tuong lai
    """

    print("\n\n")
    print("=" * 80)
    print("VERIFY: NO LOOK-AHEAD BIAS")
    print("=" * 80)

    # Load data
    m15_data = pd.read_csv('data/EURUSD_M15_30days.csv', index_col=0, parse_dates=True)

    # Test with 1000 candles
    print("\nTest 1: Backtest with first 1000 M15 candles...")
    m15_test1 = m15_data.iloc[:1000].copy()
    m15_test1['rsi'] = ta.rsi(m15_test1['close'], length=14)

    mtf1 = MultiTimeframeManager(m15_test1, base_timeframe='M15')
    mtf1.add_fvg_timeframe('H1')

    signals1 = []
    for i in range(100, len(m15_test1)):
        mtf1.update(i)
        h1_bias = mtf1.get_fvg_bias('H1', i)
        m15_rsi = m15_test1.iloc[i]['rsi']

        if h1_bias == 'BULLISH_BIAS' and m15_rsi < 30:
            signals1.append({'index': i, 'time': m15_test1.index[i]})

    print(f"  Signals: {len(signals1)}")

    # Test with 1500 candles (but only check first 1000)
    print("\nTest 2: Backtest with first 1500 M15 candles (check first 1000 only)...")
    m15_test2 = m15_data.iloc[:1500].copy()
    m15_test2['rsi'] = ta.rsi(m15_test2['close'], length=14)

    mtf2 = MultiTimeframeManager(m15_test2, base_timeframe='M15')
    mtf2.add_fvg_timeframe('H1')

    signals2 = []
    for i in range(100, len(m15_test2)):
        mtf2.update(i)

        # Only record signals from first 1000 candles
        if i < 1000:
            h1_bias = mtf2.get_fvg_bias('H1', i)
            m15_rsi = m15_test2.iloc[i]['rsi']

            if h1_bias == 'BULLISH_BIAS' and m15_rsi < 30:
                signals2.append({'index': i, 'time': m15_test2.index[i]})

    print(f"  Signals (first 1000): {len(signals2)}")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Test 1 signals: {len(signals1)}")
    print(f"Test 2 signals (first 1000): {len(signals2)}")

    if len(signals1) == len(signals2):
        # Check if all signal times match
        times1 = set(s['time'] for s in signals1)
        times2 = set(s['time'] for s in signals2)

        if times1 == times2:
            print("\n✅ PASSED: Results are IDENTICAL")
            print("   -> NO LOOK-AHEAD BIAS detected!")
        else:
            print("\n⚠️  WARNING: Signal count same but times differ")
            print(f"   Only in Test 1: {times1 - times2}")
            print(f"   Only in Test 2: {times2 - times1}")
    else:
        print("\n❌ FAILED: Results are DIFFERENT")
        print("   -> LOOK-AHEAD BIAS may exist!")


if __name__ == '__main__':
    # Run Option 1
    signals1 = option1_manual_resample_align()

    # Run Option 2
    signals2, mtf = option2_multi_timeframe_manager()

    # Verify no look-ahead bias
    verify_no_look_ahead_bias()

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
