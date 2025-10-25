"""
Strategy Parameter Optimization

Script to automatically test multiple parameter combinations
and find the best configuration for win rate and profitability.

Usage:
    python examples/optimize_strategy.py

Features:
    - Grid search across multiple parameters
    - Tests different confidence thresholds, SL/TP ratios, ADX thresholds
    - Tests different confluence weights
    - Saves all results to CSV
    - Shows top 10 best configurations

Author: Claude Code
Date: 2025-10-25
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
from datetime import datetime
from itertools import product
from strategies.fvg_confluence_strategy import FVGConfluenceStrategy
from core.backtest.backtester import Backtester
from config import DATA_DIR, BACKTEST_CONFIG
import copy


def run_single_backtest(data, config):
    """
    Run single backtest with given config

    Args:
        data: OHLCV DataFrame
        config: Backtest configuration dict

    Returns:
        dict: Performance metrics
    """
    try:
        # Initialize strategy
        strategy = FVGConfluenceStrategy(
            data=data,
            base_timeframe=config['timeframe'],
            fvg_timeframe=config['fvg_timeframe'],
            enable_adx_filter=config['enable_adx_filter'],
            min_score_threshold=config['min_confidence_score']
        )

        # Initialize backtester
        backtester = Backtester(config)

        # Run backtest
        start_idx = 100
        for i in range(start_idx, len(data)):
            current_candle = data.iloc[i]
            timestamp = data.index[i]
            high = current_candle['high']
            low = current_candle['low']
            close = current_candle['close']

            # Update open trade
            if backtester.current_trade is not None:
                backtester.update_open_trade(timestamp, high, low, close)

            # Try to open new trade
            if backtester.current_trade is None:
                analysis = strategy.analyze(i)
                if strategy.should_trade(analysis):
                    backtester.open_trade(
                        timestamp=timestamp,
                        signal_data=analysis,
                        current_price=close,
                        atr_value=analysis['atr']
                    )

        # Close any remaining trade
        if backtester.current_trade:
            final_close = data.iloc[-1]['close']
            backtester.close_trade(data.index[-1], final_close, 'END')

        # Get metrics
        metrics = backtester.get_performance_metrics()

        # Count mode distribution
        trades_df = backtester.get_trades_dataframe()
        if len(trades_df) > 0:
            virtual_count = len(trades_df[trades_df['mode'] == 'VIRTUAL'])
            real_count = len(trades_df[trades_df['mode'] == 'REAL'])
        else:
            virtual_count = 0
            real_count = 0

        return {
            'success': True,
            'total_trades': metrics['total_trades'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'total_pnl': metrics['total_pnl'],
            'return_pct': metrics['return_pct'],
            'max_drawdown': metrics['max_drawdown'],
            'avg_win': metrics['avg_win'],
            'avg_loss': metrics['avg_loss'],
            'virtual_trades': virtual_count,
            'real_trades': real_count,
            'final_balance': metrics['final_balance'],
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def optimize_confidence_and_filters():
    """
    Optimize confidence score, ADX threshold, and consecutive losses trigger

    This is the FASTEST optimization - tests filtering parameters
    """
    print("\n" + "="*100)
    print("OPTIMIZATION 1: CONFIDENCE & FILTERS")
    print("="*100)
    print("Testing different confidence thresholds, ADX thresholds, and loss triggers...")

    # Load base config
    base_config = BACKTEST_CONFIG.copy()

    # Load data
    symbol = base_config['symbol']
    timeframe = base_config['timeframe']
    days = base_config['days']
    data_file = DATA_DIR / f"{symbol}_{timeframe}_{days}days.csv"

    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        print("   Please run: python data/batch_download_mt5_data.py")
        return None

    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} candles from {symbol} {timeframe}")

    # Parameter grid
    confidence_scores = [70, 75, 80, 85, 90]
    adx_thresholds = [20, 25, 30, 35]
    loss_triggers = [1, 2, 3, 5]

    total_tests = len(confidence_scores) * len(adx_thresholds) * len(loss_triggers)
    print(f"\nTotal combinations to test: {total_tests}")
    print(f"  Confidence: {confidence_scores}")
    print(f"  ADX Threshold: {adx_thresholds}")
    print(f"  Loss Trigger: {loss_triggers}")

    results = []
    test_num = 0

    for conf, adx, trigger in product(confidence_scores, adx_thresholds, loss_triggers):
        test_num += 1

        # Create test config
        test_config = base_config.copy()
        test_config['min_confidence_score'] = conf
        test_config['adx_threshold'] = adx
        test_config['consecutive_losses_trigger'] = trigger

        print(f"\n[{test_num}/{total_tests}] Testing: Confidence={conf}%, ADX={adx}, Trigger={trigger}")

        # Run backtest
        metrics = run_single_backtest(data, test_config)

        if metrics['success']:
            result = {
                'confidence_score': conf,
                'adx_threshold': adx,
                'loss_trigger': trigger,
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'return_pct': metrics['return_pct'],
                'max_drawdown': metrics['max_drawdown'],
                'virtual_trades': metrics['virtual_trades'],
                'real_trades': metrics['real_trades'],
            }
            results.append(result)

            print(f"   Win Rate: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f} | "
                  f"Trades: {metrics['total_trades']} | Return: {metrics['return_pct']:.2f}%")
        else:
            print(f"   [ERROR] {metrics['error']}")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = DATA_DIR / f"optimization_confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\n[SAVED] Results saved to: {output_file}")

    return results_df


def optimize_sl_tp_ratios():
    """
    Optimize SL/TP multipliers

    This tests different risk/reward ratios
    """
    print("\n" + "="*100)
    print("OPTIMIZATION 2: SL/TP RATIOS")
    print("="*100)
    print("Testing different SL/TP multipliers...")

    # Load base config
    base_config = BACKTEST_CONFIG.copy()

    # Use BEST confidence settings from optimization 1 (or default to safe values)
    base_config['min_confidence_score'] = 85.0
    base_config['adx_threshold'] = 30.0
    base_config['consecutive_losses_trigger'] = 3

    # Load data
    symbol = base_config['symbol']
    timeframe = base_config['timeframe']
    days = base_config['days']
    data_file = DATA_DIR / f"{symbol}_{timeframe}_{days}days.csv"

    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        return None

    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} candles from {symbol} {timeframe}")

    # Parameter grid
    sl_multipliers = [1.0, 1.5, 2.0, 2.5]
    tp_multipliers = [2.0, 3.0, 4.0, 5.0]

    total_tests = len(sl_multipliers) * len(tp_multipliers)
    print(f"\nTotal combinations to test: {total_tests}")
    print(f"  SL Multipliers: {sl_multipliers}")
    print(f"  TP Multipliers: {tp_multipliers}")

    results = []
    test_num = 0

    for sl_mult, tp_mult in product(sl_multipliers, tp_multipliers):
        # Skip if TP <= SL (invalid R:R)
        if tp_mult <= sl_mult:
            continue

        test_num += 1

        # Create test config
        test_config = base_config.copy()
        test_config['atr_sl_multiplier'] = sl_mult
        test_config['atr_tp_multiplier'] = tp_mult

        risk_reward = tp_mult / sl_mult
        print(f"\n[{test_num}] Testing: SL={sl_mult}x ATR, TP={tp_mult}x ATR (R:R = 1:{risk_reward:.2f})")

        # Run backtest
        metrics = run_single_backtest(data, test_config)

        if metrics['success']:
            result = {
                'sl_multiplier': sl_mult,
                'tp_multiplier': tp_mult,
                'risk_reward': risk_reward,
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'return_pct': metrics['return_pct'],
                'max_drawdown': metrics['max_drawdown'],
            }
            results.append(result)

            print(f"   Win Rate: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f} | "
                  f"Trades: {metrics['total_trades']} | Return: {metrics['return_pct']:.2f}%")
        else:
            print(f"   [ERROR] {metrics['error']}")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = DATA_DIR / f"optimization_sltp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\n[SAVED] Results saved to: {output_file}")

    return results_df


def optimize_confluence_weights():
    """
    Optimize confluence weights

    This tests different weightings for FVG, VWAP, OBV, Volume
    """
    print("\n" + "="*100)
    print("OPTIMIZATION 3: CONFLUENCE WEIGHTS")
    print("="*100)
    print("Testing different confluence weightings...")

    # Load base config
    base_config = BACKTEST_CONFIG.copy()

    # Use BEST settings from previous optimizations
    base_config['min_confidence_score'] = 85.0
    base_config['adx_threshold'] = 30.0
    base_config['consecutive_losses_trigger'] = 3

    # Load data
    symbol = base_config['symbol']
    timeframe = base_config['timeframe']
    days = base_config['days']
    data_file = DATA_DIR / f"{symbol}_{timeframe}_{days}days.csv"

    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        return None

    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} candles from {symbol} {timeframe}")

    # Pre-defined weight combinations (total = 100)
    weight_combinations = [
        {'name': 'FVG Dominant', 'fvg': 60, 'vwap': 20, 'obv': 10, 'volume': 10},
        {'name': 'Balanced', 'fvg': 50, 'vwap': 20, 'obv': 15, 'volume': 15},
        {'name': 'VWAP Focus', 'fvg': 40, 'vwap': 35, 'obv': 15, 'volume': 10},
        {'name': 'Volume Focus', 'fvg': 40, 'vwap': 20, 'obv': 15, 'volume': 25},
        {'name': 'FVG + VWAP', 'fvg': 55, 'vwap': 30, 'obv': 10, 'volume': 5},
        {'name': 'FVG + Volume', 'fvg': 55, 'vwap': 15, 'obv': 5, 'volume': 25},
    ]

    total_tests = len(weight_combinations)
    print(f"\nTotal combinations to test: {total_tests}")

    results = []
    test_num = 0

    for weights in weight_combinations:
        test_num += 1

        # Create test config
        test_config = base_config.copy()
        test_config['confluence_weights'] = {
            'fvg': weights['fvg'],
            'vwap': weights['vwap'],
            'obv': weights['obv'],
            'volume': weights['volume'],
        }

        print(f"\n[{test_num}/{total_tests}] Testing: {weights['name']}")
        print(f"   FVG={weights['fvg']}% | VWAP={weights['vwap']}% | "
              f"OBV={weights['obv']}% | Volume={weights['volume']}%")

        # Run backtest
        metrics = run_single_backtest(data, test_config)

        if metrics['success']:
            result = {
                'weight_name': weights['name'],
                'fvg_weight': weights['fvg'],
                'vwap_weight': weights['vwap'],
                'obv_weight': weights['obv'],
                'volume_weight': weights['volume'],
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'return_pct': metrics['return_pct'],
                'max_drawdown': metrics['max_drawdown'],
            }
            results.append(result)

            print(f"   Win Rate: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f} | "
                  f"Trades: {metrics['total_trades']} | Return: {metrics['return_pct']:.2f}%")
        else:
            print(f"   [ERROR] {metrics['error']}")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    output_file = DATA_DIR / f"optimization_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\n[SAVED] Results saved to: {output_file}")

    return results_df


def print_top_results(results_df, optimization_name, top_n=10):
    """
    Print top N results

    Args:
        results_df: Results DataFrame
        optimization_name: Name of optimization
        top_n: Number of top results to show
    """
    print("\n" + "="*100)
    print(f"TOP {top_n} RESULTS - {optimization_name}")
    print("="*100)

    if results_df is None or len(results_df) == 0:
        print("[NO RESULTS]")
        return

    # Filter valid results (win rate > 0, profit factor > 0)
    valid_results = results_df[
        (results_df['win_rate'] > 0) &
        (results_df['profit_factor'] > 0)
    ].copy()

    if len(valid_results) == 0:
        print("[NO VALID RESULTS]")
        return

    # Sort by composite score: (win_rate * profit_factor * return_pct) / max_drawdown
    valid_results['composite_score'] = (
        valid_results['win_rate'] *
        valid_results['profit_factor'] *
        (1 + valid_results['return_pct'] / 100)
    ) / (1 + valid_results['max_drawdown'] / 100)

    # Sort by composite score
    top_results = valid_results.nlargest(top_n, 'composite_score')

    # Print results
    for idx, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"\n[{idx}] Composite Score: {row['composite_score']:.2f}")

        # Print parameters
        for col in results_df.columns:
            if col not in ['total_trades', 'win_rate', 'profit_factor', 'return_pct',
                          'max_drawdown', 'composite_score', 'virtual_trades', 'real_trades']:
                print(f"   {col}: {row[col]}")

        # Print metrics
        print(f"   Win Rate: {row['win_rate']:.1f}%")
        print(f"   Profit Factor: {row['profit_factor']:.2f}")
        print(f"   Return: {row['return_pct']:.2f}%")
        print(f"   Max DD: {row['max_drawdown']:.2f}%")
        print(f"   Total Trades: {int(row['total_trades'])}")

        if 'virtual_trades' in row:
            print(f"   Virtual/Real: {int(row['virtual_trades'])}/{int(row['real_trades'])}")


def main():
    """
    Main optimization function
    """
    print("\n" + "="*100)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("="*100)
    print("\nThis script will test multiple parameter combinations to find optimal settings.")
    print("\nOptimization Steps:")
    print("  1. Confidence & Filters (confidence score, ADX, loss trigger)")
    print("  2. SL/TP Ratios (risk/reward optimization)")
    print("  3. Confluence Weights (indicator weight optimization)")
    print("\nEach optimization will save results to CSV in the data/ folder.")
    print("\n" + "="*100)

    # Ask user which optimizations to run
    print("\nSelect optimizations to run:")
    print("  1 = Confidence & Filters (RECOMMENDED FIRST)")
    print("  2 = SL/TP Ratios")
    print("  3 = Confluence Weights")
    print("  all = Run all optimizations")

    choice = input("\nEnter choice (1/2/3/all): ").strip().lower()

    results = {}

    if choice == '1' or choice == 'all':
        results['confidence'] = optimize_confidence_and_filters()
        if results['confidence'] is not None:
            print_top_results(results['confidence'], "CONFIDENCE & FILTERS", top_n=10)

    if choice == '2' or choice == 'all':
        results['sltp'] = optimize_sl_tp_ratios()
        if results['sltp'] is not None:
            print_top_results(results['sltp'], "SL/TP RATIOS", top_n=10)

    if choice == '3' or choice == 'all':
        results['weights'] = optimize_confluence_weights()
        if results['weights'] is not None:
            print_top_results(results['weights'], "CONFLUENCE WEIGHTS", top_n=10)

    # Final summary
    print("\n" + "="*100)
    print("OPTIMIZATION COMPLETE")
    print("="*100)
    print("\nResults saved to data/ folder.")
    print("Review the CSV files and update config.py with the best parameters.")
    print("\nTo apply the best settings:")
    print("  1. Open config.py")
    print("  2. Update BACKTEST_CONFIG with top-performing parameters")
    print("  3. Run: python examples/run_backtest.py")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
