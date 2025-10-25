"""
INDICATORS + CONFLUENCE SCORING EXAMPLE

Demonstrates how to use the indicators module with FVG for confluence scoring.

Author: Claude Code
Date: 2025-10-24
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
from core.fvg.fvg_manager import FVGManager
from indicators import (
    ATRIndicator, VWAPIndicator, OBVIndicator,
    VolumeAnalyzer, ADXIndicator, ConfluenceScorer
)
from config import DATA_CONFIG, FVG_CONFIG


def example_basic_indicators():
    """
    Example 1: Basic indicator usage
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC INDICATORS")
    print("="*80)

    # Load data
    data_path = f"data/{DATA_CONFIG['symbol']}_{DATA_CONFIG['timeframe']}_{DATA_CONFIG['days']}days.csv"
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"\nLoaded {len(data)} candles from {data_path}")

    # === ATR ===
    print("\n--- ATR (Average True Range) ---")
    atr = ATRIndicator(period=14)
    atr_values = atr.calculate(data)

    print(f"Current ATR: {atr_values.iloc[-1]:.5f}")
    print(f"Average ATR: {atr_values.mean():.5f}")

    # Position sizing example
    entry_price = data.iloc[-1]['close']
    current_atr = atr_values.iloc[-1]
    lot_size = atr.get_position_size(10000, 0.01, current_atr)

    print(f"\nPosition Sizing (Account: $10,000, Risk: 1%):")
    print(f"  Lot size: {lot_size:.2f}")

    # SL/TP example
    sl_tp = atr.get_sl_tp_levels(entry_price, current_atr, 'BUY')
    print(f"\nSL/TP Levels (Entry: {entry_price:.5f}):")
    print(f"  SL: {sl_tp['sl']:.5f} (distance: {sl_tp['sl_distance']:.5f})")
    print(f"  TP: {sl_tp['tp']:.5f} (distance: {sl_tp['tp_distance']:.5f})")
    print(f"  R:R: {sl_tp['rr_ratio']:.2f}")

    # === VWAP ===
    print("\n--- VWAP (Volume Weighted Average Price) ---")
    vwap = VWAPIndicator()
    vwap_values = vwap.calculate(data)

    current_vwap = vwap_values.iloc[-1]
    current_price = data.iloc[-1]['close']

    print(f"Current VWAP: {current_vwap:.5f}")
    print(f"Current Price: {current_price:.5f}")
    print(f"Price vs VWAP: {'ABOVE (Bullish)' if current_price > current_vwap else 'BELOW (Bearish)'}")

    # === OBV ===
    print("\n--- OBV (On-Balance Volume) ---")
    obv = OBVIndicator()
    obv_data = obv.get_obv_sma(data, period=20)

    current_obv = obv_data.iloc[-1]['obv']
    current_obv_sma = obv_data.iloc[-1]['obv_sma']
    obv_above_sma = obv_data.iloc[-1]['obv_above_sma']

    print(f"Current OBV: {current_obv:.0f}")
    print(f"OBV SMA(20): {current_obv_sma:.0f}")
    print(f"OBV Trend: {'RISING (Accumulation)' if obv_above_sma else 'FALLING (Distribution)'}")

    # === Volume Analysis ===
    print("\n--- Volume Analysis ---")
    vol_analyzer = VolumeAnalyzer(period=20, spike_threshold=1.5)
    vol_analysis = vol_analyzer.calculate(data)

    current_vol = data.iloc[-1]['volume']
    avg_vol = vol_analysis.iloc[-1]['avg_volume']
    vol_ratio = vol_analysis.iloc[-1]['volume_ratio']
    is_spike = vol_analysis.iloc[-1]['is_spike']

    print(f"Current Volume: {current_vol:.0f}")
    print(f"Average Volume: {avg_vol:.0f}")
    print(f"Volume Ratio: {vol_ratio:.2f}x")
    print(f"Volume Spike: {'YES' if is_spike else 'NO'}")

    # === ADX ===
    print("\n--- ADX (Average Directional Index) ---")
    adx = ADXIndicator(period=14)
    adx_data = adx.calculate(data)

    current_adx = adx_data.iloc[-1]['adx']
    trend_strength = adx_data.iloc[-1]['trend_strength']

    print(f"Current ADX: {current_adx:.2f}")
    print(f"Trend Strength: {trend_strength}")
    print(f"Market State: {'TRENDING' if current_adx >= 25 else 'RANGING'}")


def example_confluence_scoring():
    """
    Example 2: Confluence Scoring with FVG
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 2: CONFLUENCE SCORING")
    print("="*80)

    # Load data
    data_path = f"data/{DATA_CONFIG['symbol']}_{DATA_CONFIG['timeframe']}_{DATA_CONFIG['days']}days.csv"
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Initialize FVG Manager
    fvg_manager = FVGManager(
        lookback_days=FVG_CONFIG['lookback_days'],
        min_gap_atr_ratio=FVG_CONFIG['min_gap_atr_ratio']
    )

    # Initialize ATR for FVG detection
    atr_indicator = ATRIndicator(period=14)
    atr_values = atr_indicator.calculate(data)

    # Initialize Confluence Scorer
    scorer = ConfluenceScorer(adx_enabled=True, adx_threshold=25.0)

    print(f"\nScoring Weights:")
    for component, weight in scorer.get_weights().items():
        print(f"  {component}: {weight}%")

    # Backtest with confluence scoring
    print("\n" + "="*80)
    print("BACKTESTING WITH CONFLUENCE SCORING")
    print("="*80)

    signals = []

    for i in range(100, len(data)):
        # Update FVG manager
        fvg_manager.update(data.iloc[:i+1], i, atr_values.iloc[i])

        # Get FVG structure
        fvg_structure = fvg_manager.get_market_structure(data.iloc[i]['close'])

        # Calculate confluence score
        score_result = scorer.calculate_score(
            data=data.iloc[:i+1],
            index=i,
            fvg_structure=fvg_structure,
            atr_value=atr_values.iloc[i]
        )

        # Record high/medium confidence signals
        if score_result['should_trade']:
            signals.append({
                'index': i,
                'time': data.index[i],
                'signal': score_result['signal'],
                'score': score_result['total_score'],
                'confidence': score_result['confidence'],
                'components': score_result['components'],
                'reason': score_result['reason'],
                'price': data.iloc[i]['close'],
                'sl_tp': score_result['sl_tp']
            })

            # Print signal
            print(f"\n{score_result['signal']} Signal at {data.index[i]}")
            print(f"  Score: {score_result['total_score']:.1f}/100 ({score_result['confidence']})")
            print(f"  Price: {data.iloc[i]['close']:.5f}")
            print(f"  Components:")
            for comp, value in score_result['components'].items():
                if comp != 'adx_filter':
                    print(f"    {comp}: {value:.1f}")
                else:
                    print(f"    {comp}: {value}")

            if score_result['sl_tp']:
                print(f"  SL: {score_result['sl_tp']['sl']:.5f}")
                print(f"  TP: {score_result['sl_tp']['tp']:.5f}")
                print(f"  R:R: {score_result['sl_tp']['rr_ratio']:.2f}")

    # Summary
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    print(f"Total Signals: {len(signals)}")
    print(f"  HIGH confidence: {len([s for s in signals if s['confidence'] == 'HIGH'])}")
    print(f"  MEDIUM confidence: {len([s for s in signals if s['confidence'] == 'MEDIUM'])}")
    print(f"  BUY signals: {len([s for s in signals if s['signal'] == 'BUY'])}")
    print(f"  SELL signals: {len([s for s in signals if s['signal'] == 'SELL'])}")

    # Export signals
    if signals:
        signals_df = pd.DataFrame([{
            'time': s['time'],
            'signal': s['signal'],
            'score': s['score'],
            'confidence': s['confidence'],
            'price': s['price'],
            'sl': s['sl_tp']['sl'] if s['sl_tp'] else None,
            'tp': s['sl_tp']['tp'] if s['sl_tp'] else None,
        } for s in signals])

        output_path = 'logs/confluence_signals.csv'
        signals_df.to_csv(output_path, index=False)
        print(f"\nSignals exported to: {output_path}")


def example_custom_weights():
    """
    Example 3: Custom weights configuration
    """
    print("\n\n" + "="*80)
    print("EXAMPLE 3: CUSTOM WEIGHTS")
    print("="*80)

    # Custom weights (more weight on FVG)
    custom_weights = {
        'fvg': 60,      # Increase FVG importance
        'vwap': 20,
        'obv': 10,
        'volume': 10
    }

    scorer = ConfluenceScorer(weights=custom_weights)
    print(f"\nCustom Weights:")
    for component, weight in scorer.get_weights().items():
        print(f"  {component}: {weight}%")


if __name__ == '__main__':
    # Run examples
    example_basic_indicators()
    example_confluence_scoring()
    example_custom_weights()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED!")
    print("="*80)
