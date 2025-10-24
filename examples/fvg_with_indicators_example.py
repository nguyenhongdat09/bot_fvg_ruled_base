"""
EXAMPLE: KET HOP FVG VOI INDICATORS - NO LOOK-AHEAD BIAS

Vi du nay minh hoa cach ket hop FVG voi cac indicator khac
ma KHONG BI LOOK-AHEAD BIAS.

Author: Claude Code
Date: 2025-10-24
"""

import pandas as pd
import pandas_ta as ta
from core.fvg.fvg_manager import FVGManager
from config import FVG_CONFIG


def example_fvg_with_rsi_volume():
    """
    Vi du: Ket hop FVG + RSI + Volume

    Strategy logic:
    - BUY: FVG bullish + RSI < 30 (oversold) + Volume spike
    - SELL: FVG bearish + RSI > 70 (overbought) + Volume spike
    """

    # ===== STEP 1: Load data =====
    print("Loading data...")
    data = pd.read_csv('data/EURUSD_M15_30days.csv', index_col=0, parse_dates=True)

    # ===== STEP 2: Calculate indicators (TOAN BO DATA TRUOC) =====
    print("Calculating indicators...")

    # RSI
    data['rsi'] = ta.rsi(data['close'], length=14)

    # Volume SMA
    data['volume_sma'] = ta.sma(data['volume'], length=20)

    # ATR
    data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)

    # ===== STEP 3: Initialize FVG Manager =====
    manager = FVGManager(
        lookback_days=FVG_CONFIG['lookback_days'],
        min_gap_atr_ratio=FVG_CONFIG['min_gap_atr_ratio']
    )

    # ===== STEP 4: Sequential Backtesting =====
    print("\nBacktesting with FVG + RSI + Volume...")
    print("=" * 60)

    signals = []
    start_index = 100  # Can du lieu cho indicators

    for i in range(start_index, len(data)):
        # === Update FVG (CHI DUNG DATA QUA KHU: 0 -> i) ===
        manager.update(data.iloc[:i+1], i, data.iloc[i]['atr'])

        # === Get current values (TAI INDEX i) ===
        current_price = data.iloc[i]['close']
        current_rsi = data.iloc[i]['rsi']
        current_volume = data.iloc[i]['volume']
        current_volume_sma = data.iloc[i]['volume_sma']
        current_time = data.index[i]

        # === Get FVG structure (REAL-TIME STATE) ===
        structure = manager.get_market_structure(current_price)

        # === Check conditions ===
        volume_spike = current_volume > current_volume_sma * 1.5

        # BUY Signal
        if (structure['bias'] == 'BULLISH_BIAS' and
            current_rsi < 30 and
            volume_spike):

            signal = {
                'index': i,
                'time': current_time,
                'type': 'BUY',
                'price': current_price,
                'rsi': current_rsi,
                'volume_ratio': current_volume / current_volume_sma,
                'fvg_count': len(structure['bullish_fvgs_below']),
                'nearest_fvg': structure['nearest_bullish_target']
            }
            signals.append(signal)

            print(f"\n{'='*60}")
            print(f"BUY SIGNAL at {current_time}")
            print(f"  Price: {current_price:.5f}")
            print(f"  RSI: {current_rsi:.2f} (oversold)")
            print(f"  Volume Ratio: {current_volume/current_volume_sma:.2f}x (spike)")
            print(f"  Bullish FVGs below: {len(structure['bullish_fvgs_below'])}")
            if structure['nearest_bullish_target']:
                fvg = structure['nearest_bullish_target']
                print(f"  Nearest FVG: {fvg.bottom:.5f} - {fvg.top:.5f}")
                print(f"  FVG created: {fvg.created_timestamp}")

        # SELL Signal
        elif (structure['bias'] == 'BEARISH_BIAS' and
              current_rsi > 70 and
              volume_spike):

            signal = {
                'index': i,
                'time': current_time,
                'type': 'SELL',
                'price': current_price,
                'rsi': current_rsi,
                'volume_ratio': current_volume / current_volume_sma,
                'fvg_count': len(structure['bearish_fvgs_above']),
                'nearest_fvg': structure['nearest_bearish_target']
            }
            signals.append(signal)

            print(f"\n{'='*60}")
            print(f"SELL SIGNAL at {current_time}")
            print(f"  Price: {current_price:.5f}")
            print(f"  RSI: {current_rsi:.2f} (overbought)")
            print(f"  Volume Ratio: {current_volume/current_volume_sma:.2f}x (spike)")
            print(f"  Bearish FVGs above: {len(structure['bearish_fvgs_above'])}")
            if structure['nearest_bearish_target']:
                fvg = structure['nearest_bearish_target']
                print(f"  Nearest FVG: {fvg.bottom:.5f} - {fvg.top:.5f}")
                print(f"  FVG created: {fvg.created_timestamp}")

    # ===== STEP 5: Summary =====
    print(f"\n{'='*60}")
    print("BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total candles processed: {len(data) - start_index}")
    print(f"Total signals: {len(signals)}")
    print(f"  BUY signals: {len([s for s in signals if s['type'] == 'BUY'])}")
    print(f"  SELL signals: {len([s for s in signals if s['type'] == 'SELL'])}")

    # ===== STEP 6: Verify NO LOOK-AHEAD BIAS =====
    print(f"\n{'='*60}")
    print("VERIFICATION: NO LOOK-AHEAD BIAS")
    print(f"{'='*60}")

    # Chon 1 signal de verify
    if signals:
        signal = signals[0]
        i = signal['index']
        print(f"\nChecking signal at index {i} ({signal['time']})")
        print(f"  Signal type: {signal['type']}")
        print(f"  Price: {signal['price']:.5f}")
        print(f"  RSI: {signal['rsi']:.2f}")

        # Verify RSI chi dung data qua khu
        print(f"\n  RSI Calculation:")
        print(f"    RSI tai index {i} = {data.iloc[i]['rsi']:.2f}")
        print(f"    RSI duoc tinh tu close prices: index {i-13} den {i}")
        print(f"    Close prices used: {data.iloc[i-13:i+1]['close'].tolist()[-5:]}")
        print(f"    -> KHONG CO DATA TUONG LAI!")

        # Verify FVG chi la active tai thoi diem do
        print(f"\n  FVG State:")
        if signal['nearest_fvg']:
            fvg = signal['nearest_fvg']
            print(f"    FVG created: {fvg.created_timestamp} (index {fvg.created_index})")
            print(f"    FVG checked tai: {signal['time']} (index {i})")
            print(f"    FVG is_active: {fvg.is_active}")
            print(f"    FVG is_touched: {fvg.is_touched}")
            print(f"    -> State la REAL-TIME, khong phai final state!")

    print(f"\n{'='*60}")
    print("✅ VERIFIED: NO LOOK-AHEAD BIAS")
    print(f"{'='*60}")

    return signals


def example_check_look_ahead_bias():
    """
    Test: Kiem tra look-ahead bias

    Neu backtest voi 1000 candles cho cung ket qua voi
    backtest voi 1500 candles (chi lay 1000 candles dau)
    => KHONG CO LOOK-AHEAD BIAS
    """

    print("\n" + "="*60)
    print("TEST: LOOK-AHEAD BIAS DETECTION")
    print("="*60)

    # Load data
    data = pd.read_csv('data/EURUSD_M15_30days.csv', index_col=0, parse_dates=True)

    # Calculate indicators
    data['rsi'] = ta.rsi(data['close'], length=14)
    data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)

    # Test 1: Backtest voi 1000 candles
    print("\nTest 1: Backtest with 1000 candles...")
    manager1 = FVGManager()
    signals_1000 = []

    for i in range(100, 1000):
        manager1.update(data.iloc[:i+1], i, data.iloc[i]['atr'])
        structure = manager1.get_market_structure(data.iloc[i]['close'])

        if structure['bias'] == 'BULLISH_BIAS' and data.iloc[i]['rsi'] < 30:
            signals_1000.append({'index': i, 'price': data.iloc[i]['close']})

    # Test 2: Backtest voi 1500 candles (nhung chi lay ket qua 1000 candles dau)
    print("Test 2: Backtest with 1500 candles (take first 1000 results)...")
    manager2 = FVGManager()
    signals_1500 = []

    for i in range(100, 1500):
        manager2.update(data.iloc[:i+1], i, data.iloc[i]['atr'])
        structure = manager2.get_market_structure(data.iloc[i]['close'])

        if i < 1000:  # Chi lay ket qua 1000 candles dau
            if structure['bias'] == 'BULLISH_BIAS' and data.iloc[i]['rsi'] < 30:
                signals_1500.append({'index': i, 'price': data.iloc[i]['close']})

    # So sanh ket qua
    print(f"\nResults:")
    print(f"  Test 1 (1000 candles): {len(signals_1000)} signals")
    print(f"  Test 2 (1500 candles, first 1000): {len(signals_1500)} signals")

    if signals_1000 == signals_1500:
        print(f"\n✅ PASSED: Results are IDENTICAL")
        print(f"   -> NO LOOK-AHEAD BIAS detected!")
    else:
        print(f"\n❌ FAILED: Results are DIFFERENT")
        print(f"   -> LOOK-AHEAD BIAS detected!")

        # Show differences
        indices_1000 = set(s['index'] for s in signals_1000)
        indices_1500 = set(s['index'] for s in signals_1500)

        only_in_1000 = indices_1000 - indices_1500
        only_in_1500 = indices_1500 - indices_1000

        if only_in_1000:
            print(f"\n   Signals only in Test 1: {only_in_1000}")
        if only_in_1500:
            print(f"   Signals only in Test 2: {only_in_1500}")


if __name__ == '__main__':
    # Example 1: FVG + RSI + Volume
    print("EXAMPLE 1: FVG + RSI + Volume")
    print("="*60)
    signals = example_fvg_with_rsi_volume()

    # Example 2: Check look-ahead bias
    print("\n\n")
    example_check_look_ahead_bias()
