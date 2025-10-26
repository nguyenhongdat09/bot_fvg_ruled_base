#!/usr/bin/env python3
"""
Microstructure Analysis + FVG Strategy Example

Demonstrates:
1. How to use MicrostructureAnalyzer to detect exhaustion
2. Integration with FVG system
3. Finding optimal entry points after FVG detected

Workflow:
FVG detected → Price moves away → Microstructure detects exhaustion → Entry signal

Author: Claude Code
Date: 2025-10-26
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from core.microstructure import MicrostructureAnalyzer, MicrostructureSignal
from core.fvg.fvg_manager import FVGManager
from core.fvg.fvg_detector import FVGDetector
from indicators.volatility import ATRIndicator
from config import DATA_DIR


def load_sample_data():
    """Load sample data"""
    print("Loading data...")
    data_file = DATA_DIR / "EURUSD_M15_180days.csv"

    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        print("   Please run: python data/batch_download_mt5_data.py")
        return None

    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"   Loaded {len(data)} candles from {data.index[0]} to {data.index[-1]}")

    return data


def demo_microstructure_analysis():
    """Demo: Basic microstructure analysis"""
    print("\n" + "="*80)
    print("DEMO 1: BASIC MICROSTRUCTURE ANALYSIS")
    print("="*80)

    # Load data
    data = load_sample_data()
    if data is None:
        return

    # Calculate ATR
    print("\nCalculating ATR...")
    atr_indicator = ATRIndicator(period=14)
    atr_series = atr_indicator.calculate(data)

    # Initialize Microstructure Analyzer
    print("\nInitializing Microstructure Analyzer...")
    analyzer = MicrostructureAnalyzer(
        cpd_method='cusum',
        cpd_sensitivity=2.0,
        hurst_window=100,
        vol_ma_period=20,
        swing_fractal_period=5,
        swing_zscore_threshold=1.5,
        entropy_window=20
    )

    # Analyze recent period
    print("\nAnalyzing recent 100 candles...")
    start_idx = len(data) - 100
    signals = []

    for i in range(start_idx, len(data)):
        signal = analyzer.analyze(
            data.iloc[:i+1],
            i,
            atr_series=atr_series.iloc[:i+1]
        )

        if signal:
            signals.append(signal)

    # Display results
    print(f"\n{'='*80}")
    print(f"RESULTS: Found {len(signals)} microstructure signals")
    print(f"{'='*80}")

    for sig in signals[-5:]:  # Show last 5 signals
        print(f"\n[{sig.timestamp}] {sig.signal_type} - {sig.direction}")
        print(f"   Price: {sig.price:.5f}")
        print(f"   Confidence: {sig.confidence:.2f}")
        print(f"   Components:")
        print(f"      Volume Exhaustion: {sig.components['volume_exhaustion'].get('score', 0):.2f}")
        print(f"      Hurst: {sig.components['hurst'].get('score', 0):.2f}")
        print(f"      Entropy: {sig.components['entropy'].get('score', 0):.2f}")
        print(f"      Swing: {sig.components['swing'].get('score', 0):.2f}")


def demo_fvg_with_microstructure():
    """Demo: FVG + Microstructure combined"""
    print("\n" + "="*80)
    print("DEMO 2: FVG + MICROSTRUCTURE COMBINED")
    print("="*80)

    # Load data
    data = load_sample_data()
    if data is None:
        return

    # Calculate ATR
    print("\nCalculating ATR...")
    atr_indicator = ATRIndicator(period=14)
    atr_series = atr_indicator.calculate(data)

    # Initialize FVG Manager
    print("\nInitializing FVG Manager...")
    fvg_manager = FVGManager(
        lookback_days=90,
        min_gap_atr_ratio=0.3
    )

    # Initialize Microstructure Analyzer
    print("Initializing Microstructure Analyzer...")
    microstructure = MicrostructureAnalyzer()

    # Process data
    print("\nProcessing data...")
    trades = []

    for i in range(100, len(data)):
        # Update FVG
        fvg_manager.update(data.iloc[:i+1], i, atr_series.iloc[i])

        # Get FVG structure
        fvg_structure = fvg_manager.get_market_structure(data.iloc[i]['close'])

        # Check if we have FVG bias
        if fvg_structure['bias'] in ['BULLISH_BIAS', 'BEARISH_BIAS']:
            # Analyze microstructure
            signal = microstructure.analyze(
                data.iloc[:i+1],
                i,
                fvg_info=fvg_structure,
                atr_series=atr_series.iloc[:i+1]
            )

            # Check if microstructure confirms FVG direction
            if signal and signal.confidence > 0.65:
                fvg_direction = 'BULLISH' if fvg_structure['bias'] == 'BULLISH_BIAS' else 'BEARISH'

                # If microstructure and FVG agree
                if signal.direction == fvg_direction:
                    trades.append({
                        'timestamp': data.index[i],
                        'direction': signal.direction,
                        'price': data.iloc[i]['close'],
                        'fvg_bias': fvg_structure['bias'],
                        'microstructure_confidence': signal.confidence,
                        'signal_type': signal.signal_type,
                        'fvg_target': fvg_structure.get(f"nearest_{fvg_direction.lower()}_target")
                    })

    # Display results
    print(f"\n{'='*80}")
    print(f"RESULTS: Found {len(trades)} high-quality trade setups")
    print(f"{'='*80}")

    for trade in trades[-10:]:  # Show last 10 trades
        print(f"\n[{trade['timestamp']}] {trade['direction']} Trade")
        print(f"   Entry Price: {trade['price']:.5f}")
        print(f"   FVG Bias: {trade['fvg_bias']}")
        print(f"   Microstructure Confidence: {trade['microstructure_confidence']:.2f}")
        print(f"   Signal Type: {trade['signal_type']}")

        if trade['fvg_target']:
            fvg = trade['fvg_target']
            print(f"   FVG Target: {fvg.top:.5f} - {fvg.bottom:.5f}")


def demo_exhaustion_detection():
    """Demo: Pure exhaustion detection"""
    print("\n" + "="*80)
    print("DEMO 3: EXHAUSTION DETECTION")
    print("="*80)

    # Load data
    data = load_sample_data()
    if data is None:
        return

    # Initialize Microstructure Analyzer
    print("\nInitializing Microstructure Analyzer...")
    analyzer = MicrostructureAnalyzer()

    # Analyze recent period
    print("\nScanning for exhaustion signals...")
    start_idx = len(data) - 200

    exhaustion_points = []

    for i in range(start_idx, len(data)):
        # Get comprehensive analysis
        analysis = analyzer.get_comprehensive_analysis(data.iloc[:i+1], i)

        # Check for exhaustion
        vol_exhaustion = analysis['volume_exhaustion']
        hurst = analysis['hurst']
        entropy = analysis['entropy']

        # Exhaustion criteria
        is_exhausted = (
            vol_exhaustion.get('is_exhausted', False) or
            hurst.get('exhaustion_score', 0) > 0.6 or
            entropy.get('is_chaotic', False)
        )

        if is_exhausted:
            exhaustion_points.append({
                'timestamp': data.index[i],
                'price': data.iloc[i]['close'],
                'vol_exhaustion': vol_exhaustion.get('exhaustion_score', 0),
                'hurst_exhaustion': hurst.get('exhaustion_score', 0),
                'entropy_score': entropy.get('entropy_score', 0),
                'vol_direction': vol_exhaustion.get('direction', 'NEUTRAL'),
                'hurst_regime': hurst.get('regime', 'UNKNOWN'),
                'entropy_regime': entropy.get('regime', 'UNKNOWN')
            })

    # Display results
    print(f"\n{'='*80}")
    print(f"RESULTS: Found {len(exhaustion_points)} exhaustion points")
    print(f"{'='*80}")

    for point in exhaustion_points[-15:]:  # Show last 15
        print(f"\n[{point['timestamp']}] EXHAUSTION DETECTED")
        print(f"   Price: {point['price']:.5f}")
        print(f"   Volume Exhaustion: {point['vol_exhaustion']:.2f} ({point['vol_direction']})")
        print(f"   Hurst Exhaustion: {point['hurst_exhaustion']:.2f} ({point['hurst_regime']})")
        print(f"   Entropy: {point['entropy_score']:.2f} ({point['entropy_regime']})")


def demo_comprehensive_report():
    """Demo: Comprehensive analysis report"""
    print("\n" + "="*80)
    print("DEMO 4: COMPREHENSIVE ANALYSIS REPORT")
    print("="*80)

    # Load data
    data = load_sample_data()
    if data is None:
        return

    # Initialize analyzer
    analyzer = MicrostructureAnalyzer()

    # Analyze current market state
    current_idx = len(data) - 1
    print(f"\nAnalyzing current market state at {data.index[current_idx]}...")

    report = analyzer.get_comprehensive_analysis(data, current_idx)

    # Display report
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MARKET ANALYSIS")
    print(f"{'='*80}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Price: {report['price']:.5f}")

    if report['signal']:
        sig = report['signal']
        print(f"\n[SIGNAL DETECTED]")
        print(f"   Type: {sig.signal_type}")
        print(f"   Direction: {sig.direction}")
        print(f"   Confidence: {sig.confidence:.2f}")
    else:
        print(f"\n[NO SIGNAL]")

    print(f"\n[COMPONENT ANALYSIS]")

    # Change Point
    cpd = report['change_point']
    print(f"\n1. Change Point Detection:")
    print(f"   Detected: {cpd.get('detected', False)}")
    if cpd.get('detected'):
        print(f"   Direction: {cpd.get('direction', 'UNKNOWN')}")
        print(f"   Score: {cpd.get('score', 0):.2f}")

    # Hurst
    hurst = report['hurst']
    print(f"\n2. Hurst Exponent:")
    print(f"   Value: {hurst.get('hurst_value', 0):.3f}")
    print(f"   Regime: {hurst.get('regime', 'UNKNOWN')}")
    print(f"   Exhaustion Score: {hurst.get('exhaustion_score', 0):.2f}")

    # Volume Exhaustion
    vol_exh = report['volume_exhaustion']
    print(f"\n3. Volume Exhaustion:")
    print(f"   Score: {vol_exh.get('exhaustion_score', 0):.2f}")
    print(f"   Direction: {vol_exh.get('direction', 'NEUTRAL')}")
    print(f"   Is Exhausted: {vol_exh.get('is_exhausted', False)}")

    # Swing
    swing = report['swing']
    print(f"\n4. Statistical Swing:")
    print(f"   Detected: {swing.get('swing_detected', False)}")
    if swing.get('swing_detected'):
        print(f"   Type: {swing.get('swing_type', 'UNKNOWN')}")
        print(f"   Price: {swing.get('swing_price', 0):.5f}")
        print(f"   Strength: {swing.get('swing_strength', 0):.2f}")

    # Entropy
    entropy = report['entropy']
    print(f"\n5. Entropy Analysis:")
    print(f"   Score: {entropy.get('entropy_score', 0):.2f}")
    print(f"   Regime: {entropy.get('regime', 'UNKNOWN')}")
    print(f"   Is Chaotic: {entropy.get('is_chaotic', False)}")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("MICROSTRUCTURE ANALYSIS + FVG INTEGRATION EXAMPLES")
    print("="*80)

    # Run demos
    demo_microstructure_analysis()
    demo_fvg_with_microstructure()
    demo_exhaustion_detection()
    demo_comprehensive_report()

    print("\n" + "="*80)
    print("ALL DEMOS COMPLETED!")
    print("="*80)


if __name__ == '__main__':
    main()
