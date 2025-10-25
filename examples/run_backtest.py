"""
Run FVG + Confluence Backtest

Script de chay backtest chien luoc FVG + Confluence.
Ket hop Strategy voi Backtester de test tren du lieu that.

Usage:
    python examples/run_backtest.py

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
from strategies.fvg_confluence_strategy import FVGConfluenceStrategy
from core.backtest.backtester import Backtester, TradeMode
from config import DATA_DIR, BACKTEST_CONFIG


def run_backtest():
    """
    Run backtest using BACKTEST_CONFIG from config.py

    IMPORTANT: To change settings, edit config.py -> BACKTEST_CONFIG

    Returns:
        tuple: (backtester, strategy, results_df)
    """
    # Get config from config.py
    cfg = BACKTEST_CONFIG

    print("\n" + "="*100)
    print("FVG + CONFLUENCE BACKTEST")
    print("="*100)

    # 1. Load data
    print("\n[STEP 1] Loading Data...")
    symbol = cfg['symbol']
    timeframe = cfg['timeframe']
    days = cfg['days']
    data_file = DATA_DIR / f"{symbol}_{timeframe}_{days}days.csv"

    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        print("\nPlease download data first:")
        print("   python data/batch_download_mt5_data.py")
        return None, None, None

    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"OK Loaded {len(data)} candles")
    print(f"   Symbol: {symbol}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Date Range: {data.index[0]} to {data.index[-1]}")

    # 2. Initialize Strategy
    print("\n[STEP 2] Initializing Strategy...")
    strategy = FVGConfluenceStrategy(
        data=data,
        base_timeframe=cfg['timeframe'],
        fvg_timeframe=cfg['fvg_timeframe'],
        enable_adx_filter=cfg['enable_adx_filter'],
        adx_threshold=cfg['adx_threshold'],  # FIXED: Pass ADX threshold!
        min_score_threshold=cfg['min_confidence_score'],
        confluence_weights=cfg.get('confluence_weights')  # FIXED: Pass weights!
    )

    # 3. Initialize Backtester
    print("\n[STEP 3] Initializing Backtester...")
    # Use config directly from config.py
    backtester = Backtester(cfg)
    print(f"OK Backtester ready")
    print(f"   Initial Balance: ${cfg['initial_balance']:,.2f}")
    print(f"   Risk per Trade: {cfg['risk_per_trade'] * 100}%")
    print(f"   Base Lot Size: {cfg['base_lot_size']}")
    print(f"   Commission: ${cfg['commission_per_lot']}/lot")
    print(f"   Martingale Trigger: {cfg['consecutive_losses_trigger']} losses")
    print(f"   Martingale Multiplier: {cfg['martingale_multiplier']}x")

    # 4. Run Backtest
    print("\n[STEP 4] Running Backtest...")
    print("="*100)

    start_idx = 100  # Skip first 100 candles for indicator warmup
    total_candles = len(data) - start_idx
    processed = 0
    last_percent = 0

    for i in range(start_idx, len(data)):
        # Update progress
        processed += 1
        percent = int(processed / total_candles * 100)
        if percent % 10 == 0 and percent != last_percent:
            last_percent = percent
            print(f"Progress: {percent}% ({processed}/{total_candles} candles)")

        # Get current candle
        current_candle = data.iloc[i]
        timestamp = data.index[i]
        high = current_candle['high']
        low = current_candle['low']
        close = current_candle['close']

        # Update open trade (check SL/TP)
        if backtester.current_trade is not None:
            closed_trade = backtester.update_open_trade(timestamp, high, low, close)

            if closed_trade:
                # Trade closed
                win_loss = "WIN" if closed_trade.is_win() else "LOSS"
                print(f"\n{'[+]' if closed_trade.is_win() else '[-]'} Trade #{len(backtester.trades)} CLOSED - {win_loss}")
                print(f"   {closed_trade.direction} @ {closed_trade.entry_price:.5f} -> {closed_trade.exit_price:.5f}")
                print(f"   Exit: {closed_trade.exit_reason}")
                print(f"   PnL: ${closed_trade.pnl:,.2f} ({closed_trade.pnl_pips:.1f} pips)")
                print(f"   Lot Size: {closed_trade.lot_size} ({closed_trade.mode.value})")
                print(f"   Balance: ${backtester.balance:,.2f}")

                # Show mode change
                if backtester.consecutive_losses >= cfg['consecutive_losses_trigger']:
                    print(f"   WARNING Consecutive Losses: {backtester.consecutive_losses}")
                    if backtester.mode == TradeMode.REAL:
                        print(f"   [!] Mode: REAL (Martingale active)")
                        print(f"   [!] Next Lot Size: {backtester.current_lot_size:.2f}")

        # Try to open new trade
        if backtester.current_trade is None:
            # Analyze market
            analysis = strategy.analyze(i)

            # Check if should trade
            if strategy.should_trade(analysis):
                trade = backtester.open_trade(
                    timestamp=timestamp,
                    signal_data=analysis,
                    current_price=close,
                    atr_value=analysis['atr']
                )

                if trade:
                    print(f"\n{'='*100}")
                    print(f"[TRADE] #{backtester.total_trades} OPENED - {trade.direction}")
                    print(f"{'='*100}")
                    print(f"   Time: {timestamp}")
                    print(f"   Price: {trade.entry_price:.5f}")
                    print(f"   Lot Size: {trade.lot_size} ({trade.mode.value})")
                    print(f"   SL: {trade.sl_price:.5f}")
                    print(f"   TP: {trade.tp_price:.5f}")
                    print(f"   Confluence Score: {trade.confluence_score:.1f}% ({trade.confidence})")
                    print(f"   FVG Bias: {trade.fvg_bias}")
                    print(f"   Balance: ${backtester.balance:,.2f}")

    # Close any remaining open trade at the end
    if backtester.current_trade:
        print("\nWARNING Closing remaining open trade at end of data...")
        final_close = data.iloc[-1]['close']
        backtester.close_trade(data.index[-1], final_close, 'END')

    # 5. Print Results
    print("\n" + "="*100)
    print("BACKTEST COMPLETED")
    print("="*100)

    backtester.print_summary()

    # 6. Get trades DataFrame
    results_df = backtester.get_trades_dataframe()

    if len(results_df) > 0:
        # Save results
        output_file = DATA_DIR / f"backtest_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Print sample trades
        print(f"\nSample Trades (first 5):")
        print(results_df.head().to_string())

    return backtester, strategy, results_df


def main():
    """
    Main function

    IMPORTANT: All config now in config.py -> BACKTEST_CONFIG
    To change settings, edit config.py directly!
    """
    cfg = BACKTEST_CONFIG

    print("\n" + "="*100)
    print("BACKTEST CONFIGURATION (from config.py)")
    print("="*100)
    print(f"Symbol: {cfg['symbol']}")
    print(f"Timeframe: {cfg['timeframe']} (FVG on {cfg['fvg_timeframe']})")
    print(f"Data Period: {cfg['days']} days")
    print(f"Initial Balance: ${cfg['initial_balance']:,.2f}")
    print(f"Risk per Trade: {cfg['risk_per_trade'] * 100}%")
    print(f"Commission: ${cfg['commission_per_lot']}/lot")
    print(f"Min Confidence Score: {cfg['min_confidence_score']}%")
    print(f"ADX Filter: {'Enabled' if cfg['enable_adx_filter'] else 'Disabled'}")
    print(f"\nTo change these settings, edit: config.py -> BACKTEST_CONFIG")
    print("="*100)

    # Run backtest (uses config from config.py)
    backtester, strategy, results = run_backtest()

    if backtester is None:
        print("\nError: Backtest failed!")
        return 1

    # Additional analysis
    if len(results) > 0:
        print("\n" + "="*100)
        print("ADDITIONAL ANALYSIS")
        print("="*100)

        # Virtual vs Real mode trades
        virtual_trades = results[results['mode'] == 'VIRTUAL']
        real_trades = results[results['mode'] == 'REAL']

        print(f"\n[MODE] Mode Distribution:")
        print(f"   VIRTUAL: {len(virtual_trades)} trades")
        print(f"   REAL: {len(real_trades)} trades")

        if len(real_trades) > 0:
            real_pnl = real_trades['pnl'].sum()
            print(f"\n[REAL] Real Mode Performance:")
            print(f"   Total PnL: ${real_pnl:,.2f}")
            print(f"   Avg PnL: ${real_trades['pnl'].mean():,.2f}")
            print(f"   Win Rate: {(real_trades['pnl'] > 0).sum() / len(real_trades) * 100:.1f}%")

        # Confidence breakdown
        print(f"\n[CONF] Confidence Breakdown:")
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            conf_trades = results[results['confidence'] == conf]
            if len(conf_trades) > 0:
                conf_win_rate = (conf_trades['pnl'] > 0).sum() / len(conf_trades) * 100
                print(f"   {conf}: {len(conf_trades)} trades, Win Rate: {conf_win_rate:.1f}%")

    print("\n" + "="*100)
    print("DONE")
    print("="*100 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())