#!/usr/bin/env python3
"""
Example Backtest Script

Demonstrates how to run a backtest with:
- Fixed SL/TP in pips
- ATR-based SL/TP
- FVG detection and signal generation
- Performance analysis

Usage:
    python examples/run_backtest.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BACKTEST_CONFIG
from core.backtest.backtester import Backtester
from core.fvg.fvg_manager import FVGManager, validate_signal_with_fvg, get_fvg_target


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


def create_sample_data(n_candles: int = 1000) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing
    
    Args:
        n_candles: Number of candles
        
    Returns:
        pd.DataFrame: OHLCV data
    """
    print(f"\n{'='*60}")
    print("Creating sample data...")
    print(f"{'='*60}")
    
    # Create timestamp index (M15 timeframe)
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_candles, freq='15min')
    
    # Generate price data with trend + noise
    np.random.seed(42)
    
    base_price = 1.10000
    trend = np.linspace(0, 0.01, n_candles)  # Uptrend 1000 pips
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


def generate_simple_signals(data: pd.DataFrame, fvg_manager: FVGManager) -> pd.DataFrame:
    """
    Generate simple trading signals based on FVG structure and price action
    
    Args:
        data: OHLC data
        fvg_manager: FVG Manager instance
        
    Returns:
        DataFrame with signals
    """
    signals = []
    
    # Calculate simple moving average for basic trend
    data['sma_20'] = data['close'].rolling(window=20).mean()
    
    for i in range(20, len(data)):  # Start after SMA warmup
        current_price = data['close'].iloc[i]
        structure = fvg_manager.get_market_structure(current_price)
        
        signal = None
        
        # Strategy 1: FVG-based signals
        if structure['bias'] == 'BULLISH_BIAS':
            if validate_signal_with_fvg(structure, 'BUY'):
                signal = 'BUY'
        elif structure['bias'] == 'BEARISH_BIAS':
            if validate_signal_with_fvg(structure, 'SELL'):
                signal = 'SELL'
        elif structure['bias'] == 'BOTH_FVG':
            # When both FVGs exist, use simple price action
            if structure['nearest_bullish_target'] and structure['nearest_bearish_target']:
                bullish_dist = structure['nearest_bullish_target'].get_distance_to_price(current_price)
                bearish_dist = structure['nearest_bearish_target'].get_distance_to_price(current_price)
                if bullish_dist < bearish_dist:
                    signal = 'BUY'
                else:
                    signal = 'SELL'
        
        # Strategy 2: If no FVG signals, use simple trend-following
        # This ensures we have some trades for testing
        if signal is None and i > 20:
            # Every 50 bars, check for trend signal
            if i % 50 == 0:
                sma = data['sma_20'].iloc[i]
                prev_close = data['close'].iloc[i-1]
                
                # BUY if price crosses above SMA
                if prev_close < sma and current_price > sma:
                    signal = 'BUY'
                # SELL if price crosses below SMA
                elif prev_close > sma and current_price < sma:
                    signal = 'SELL'
        
        signals.append({
            'timestamp': data.index[i],
            'index': i,
            'signal': signal,
            'price': current_price
        })
    
    return pd.DataFrame(signals)


def run_backtest(config: dict, mode: str = 'fixed'):
    """
    Run backtest with specified configuration
    
    Args:
        config: BACKTEST_CONFIG dictionary
        mode: 'fixed' or 'atr'
    """
    print(f"\n{'='*60}")
    print(f"RUNNING BACKTEST - {mode.upper()} MODE")
    print(f"{'='*60}")
    
    # Override mode in config
    config = config.copy()
    config['sl_tp_mode'] = mode
    
    # Create sample data
    data = create_sample_data(n_candles=1000)
    
    # Calculate ATR
    atr_period = config.get('atr_period', 14)
    atr = calculate_atr(data, period=atr_period)
    
    # Initialize FVG Manager
    print(f"\n{'='*60}")
    print("Initializing FVG Manager...")
    print(f"{'='*60}")
    
    fvg_manager = FVGManager(
        lookback_days=config['fvg_lookback_days'],
        min_gap_atr_ratio=config['fvg_min_gap_atr_ratio'],
        min_gap_pips=config['fvg_min_gap_pips']
    )
    
    # Update FVG manager with historical data
    for i in range(20, len(data)):
        fvg_manager.update(data.iloc[:i+1], i, atr.iloc[i])
    
    fvg_stats = fvg_manager.get_statistics()
    print(f"\n✓ FVG Manager initialized")
    print(f"  Active FVGs: {fvg_stats['total_active']}")
    print(f"  Total Created: {fvg_stats['total_bullish_created'] + fvg_stats['total_bearish_created']}")
    
    # Generate signals
    print(f"\n{'='*60}")
    print("Generating signals...")
    print(f"{'='*60}")
    
    signals_df = generate_simple_signals(data, fvg_manager)
    signal_count = signals_df[signals_df['signal'].notna()].shape[0]
    print(f"✓ Generated {signal_count} signals")
    
    # Initialize backtester
    print(f"\n{'='*60}")
    print("Running backtest...")
    print(f"{'='*60}")
    
    backtester = Backtester(config)
    
    # Print configuration
    if mode == 'fixed':
        print(f"\nSL/TP Configuration:")
        print(f"  Mode: Fixed Pips")
        print(f"  SL: {config['fixed_sl_pips']} pips")
        print(f"  TP: {config['fixed_tp_pips']} pips")
    else:
        print(f"\nSL/TP Configuration:")
        print(f"  Mode: ATR-based")
        print(f"  SL: ATR × {config['atr_sl_multiplier']}")
        print(f"  TP: ATR × {config['atr_tp_multiplier']}")
    
    print(f"\nRisk Management:")
    print(f"  Initial Capital: ${config['initial_capital']:,.2f}")
    print(f"  Risk per Trade: {config['risk_per_trade_percent']}%")
    print(f"  Spread: {config['spread_pips']} pips")
    
    # Run backtest loop
    trade_count = 0
    for idx, row in signals_df.iterrows():
        i = row['index']
        
        # Update equity
        backtester.update_equity(data.index[i])
        
        # Check exit conditions for open trades
        for trade in list(backtester.open_trades):
            backtester.check_exit(
                trade,
                high=data['high'].iloc[i],
                low=data['low'].iloc[i],
                close=data['close'].iloc[i],
                timestamp=data.index[i],
                index=i
            )
        
        # Check for new signal
        if pd.notna(row['signal']) and row['signal'] in ['BUY', 'SELL']:
            # Get ATR value if needed
            atr_value = atr.iloc[i] if mode == 'atr' else None
            
            # Open trade
            trade = backtester.open_trade(
                signal=row['signal'],
                entry_price=data['close'].iloc[i],
                entry_time=data.index[i],
                entry_index=i,
                atr_value=atr_value
            )
            
            if trade:
                trade_count += 1
                if trade_count <= 5:  # Print first 5 trades
                    print(f"\n  Trade #{trade.trade_id}:")
                    print(f"    Signal: {trade.signal}")
                    print(f"    Entry: {trade.entry_price:.5f} @ {trade.entry_time}")
                    print(f"    SL: {trade.sl_price:.5f}")
                    print(f"    TP: {trade.tp_price:.5f}")
                    print(f"    Lot Size: {trade.lot_size}")
    
    # Close all remaining trades
    backtester.close_all_trades(
        data['close'].iloc[-1],
        data.index[-1],
        len(data) - 1
    )
    
    # Get performance metrics
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    
    metrics = backtester.get_performance_metrics()
    
    print(f"\nPerformance Metrics:")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Winning Trades: {metrics['winning_trades']}")
    print(f"  Losing Trades: {metrics['losing_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2f}%")
    print(f"\nP&L:")
    print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"  Average Win: ${metrics['avg_win']:,.2f}")
    print(f"  Average Loss: ${metrics['avg_loss']:,.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"\nCapital:")
    print(f"  Initial Capital: ${metrics['initial_capital']:,.2f}")
    print(f"  Final Capital: ${metrics['final_capital']:,.2f}")
    print(f"  Max Drawdown: ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2f}%)")
    
    # Export trades
    trades_df = backtester.get_trades_dataframe()
    if not trades_df.empty:
        os.makedirs('logs', exist_ok=True)
        output_file = f'logs/backtest_trades_{mode}.csv'
        trades_df.to_csv(output_file, index=False)
        print(f"\n✓ Trades exported to: {output_file}")
    
    return backtester, metrics


def main():
    """Main function"""
    print("\n" + "="*60)
    print("FVG BACKTEST - SL/TP TESTING")
    print("="*60)
    
    # Test both modes
    modes = ['fixed', 'atr']
    
    results = {}
    
    for mode in modes:
        backtester, metrics = run_backtest(BACKTEST_CONFIG, mode=mode)
        results[mode] = metrics
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON: FIXED vs ATR")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<25} {'Fixed':<15} {'ATR':<15}")
    print("-" * 60)
    print(f"{'Total Trades':<25} {results['fixed']['total_trades']:<15} {results['atr']['total_trades']:<15}")
    print(f"{'Win Rate %':<25} {results['fixed']['win_rate']:<15.2f} {results['atr']['win_rate']:<15.2f}")
    print(f"{'Total Return %':<25} {results['fixed']['total_return_pct']:<15.2f} {results['atr']['total_return_pct']:<15.2f}")
    print(f"{'Profit Factor':<25} {results['fixed']['profit_factor']:<15.2f} {results['atr']['profit_factor']:<15.2f}")
    print(f"{'Max Drawdown %':<25} {results['fixed']['max_drawdown_pct']:<15.2f} {results['atr']['max_drawdown_pct']:<15.2f}")
    
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE!")
    print(f"{'='*60}")
    
    print("\n📊 Output files:")
    print("  - logs/backtest_trades_fixed.csv")
    print("  - logs/backtest_trades_atr.csv")
    
    print("\n💡 Tips:")
    print("  - Adjust SL/TP in config.py BACKTEST_CONFIG")
    print("  - Toggle between 'fixed' and 'atr' modes")
    print("  - Modify fixed_sl_pips and fixed_tp_pips for fixed mode")
    print("  - Modify atr_sl_multiplier and atr_tp_multiplier for ATR mode")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
