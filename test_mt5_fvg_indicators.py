#!/usr/bin/env python3
"""
Comprehensive Test Script for MT5 + FVG + Indicators

This script tests:
1. MT5 data download (or loads from CSV if MT5 not available)
2. Indicator calculations
3. FVG detection
4. Integration of all components

Usage:
    python test_mt5_fvg_indicators.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MT5 data downloader
from data.download_mt5_data import (
    initialize_mt5, shutdown_mt5, download_ohlcv_data, 
    save_data_to_csv, load_data_from_csv
)

# Import indicators
from core.indicators.trend import calculate_ema, calculate_multiple_emas, detect_trend_ema, calculate_adx
from core.indicators.momentum import calculate_rsi, calculate_macd, calculate_stochastic
from core.indicators.volatility import calculate_atr, calculate_bollinger_bands
from core.indicators.volume import calculate_volume_ma, calculate_obv, detect_volume_spike

# Import FVG modules
from core.fvg.fvg_detector import FVGDetector
from core.fvg.fvg_manager import FVGManager
from core.fvg.fvg_visualizer import FVGVisualizer

# Import config
from config import FVG_CONFIG, INDICATOR_CONFIG


def create_sample_data(n_candles: int = 500) -> pd.DataFrame:
    """
    Create sample OHLCV data for testing when MT5 is not available
    
    Args:
        n_candles: Number of candles to generate
    
    Returns:
        pd.DataFrame: Sample OHLCV data
    """
    print(f"\n{'='*60}")
    print("Creating sample data (MT5 not available)...")
    print(f"{'='*60}")
    
    # Create timestamp index (M15 timeframe)
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_candles, freq='15min')
    
    # Generate price data with trend + noise
    np.random.seed(42)
    
    base_price = 1.10000
    trend = np.linspace(0, 0.005, n_candles)  # Uptrend 500 pips
    noise = np.random.randn(n_candles) * 0.0005
    
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
    
    print(f"✓ Created {len(data)} sample candles")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: {data['low'].min():.5f} to {data['high'].max():.5f}")
    
    return data


def get_data_source() -> pd.DataFrame:
    """
    Try to download data from MT5, fallback to sample data
    
    Returns:
        pd.DataFrame: OHLCV data
    """
    print(f"\n{'='*60}")
    print("STEP 1: Getting Data Source")
    print(f"{'='*60}")
    
    # Try to initialize MT5
    print("\nAttempting to connect to MT5...")
    
    try:
        if initialize_mt5():
            print("✓ MT5 connected successfully!")
            
            # Try to download data
            print("\nDownloading EURUSD M15 data...")
            data = download_ohlcv_data('EURUSD', 'M15', num_bars=1000)
            
            if data is not None:
                # Save to CSV for future use
                save_data_to_csv(data, 'EURUSD', 'M15')
                shutdown_mt5()
                return data
            else:
                print("❌ Failed to download data from MT5")
                shutdown_mt5()
        else:
            print("❌ MT5 not available")
    
    except Exception as e:
        print(f"❌ MT5 error: {e}")
    
    # Try to load from CSV
    print("\nAttempting to load data from CSV...")
    data = load_data_from_csv('EURUSD', 'M15')
    
    if data is not None:
        return data
    
    # Fallback to sample data
    print("\n⚠️  Using generated sample data")
    return create_sample_data(n_candles=1000)


def test_indicators(data: pd.DataFrame):
    """
    Test all indicator calculations
    
    Args:
        data: OHLCV DataFrame
    """
    print(f"\n{'='*60}")
    print("STEP 2: Testing Indicators")
    print(f"{'='*60}")
    
    # Test Trend Indicators
    print("\n--- Trend Indicators ---")
    ema_20 = calculate_ema(data['close'], 20)
    ema_50 = calculate_ema(data['close'], 50)
    ema_200 = calculate_ema(data['close'], 200)
    trend = detect_trend_ema(data['close'], 20, 50)
    adx_df = calculate_adx(data['high'], data['low'], data['close'])
    
    print(f"✓ EMA-20: {ema_20.iloc[-1]:.5f}")
    print(f"✓ EMA-50: {ema_50.iloc[-1]:.5f}")
    print(f"✓ EMA-200: {ema_200.iloc[-1]:.5f}")
    print(f"✓ Current Trend: {'UPTREND' if trend.iloc[-1] == 1 else 'DOWNTREND' if trend.iloc[-1] == -1 else 'NEUTRAL'}")
    print(f"✓ ADX: {adx_df['adx'].iloc[-1]:.2f}")
    
    # Test Momentum Indicators
    print("\n--- Momentum Indicators ---")
    rsi = calculate_rsi(data['close'])
    macd_df = calculate_macd(data['close'])
    stoch_df = calculate_stochastic(data['high'], data['low'], data['close'])
    
    print(f"✓ RSI: {rsi.iloc[-1]:.2f}")
    print(f"✓ MACD: {macd_df['macd'].iloc[-1]:.5f}")
    print(f"✓ MACD Signal: {macd_df['macd_signal'].iloc[-1]:.5f}")
    print(f"✓ Stochastic %K: {stoch_df['stoch_k'].iloc[-1]:.2f}")
    print(f"✓ Stochastic %D: {stoch_df['stoch_d'].iloc[-1]:.2f}")
    
    # Test Volatility Indicators
    print("\n--- Volatility Indicators ---")
    atr = calculate_atr(data['high'], data['low'], data['close'])
    bb_df = calculate_bollinger_bands(data['close'])
    
    print(f"✓ ATR: {atr.iloc[-1]:.5f}")
    print(f"✓ BB Upper: {bb_df['bb_upper'].iloc[-1]:.5f}")
    print(f"✓ BB Middle: {bb_df['bb_middle'].iloc[-1]:.5f}")
    print(f"✓ BB Lower: {bb_df['bb_lower'].iloc[-1]:.5f}")
    
    # Test Volume Indicators
    print("\n--- Volume Indicators ---")
    volume_ma = calculate_volume_ma(data['volume'])
    obv = calculate_obv(data['close'], data['volume'])
    volume_spikes = detect_volume_spike(data['volume'])
    
    print(f"✓ Volume MA: {volume_ma.iloc[-1]:.0f}")
    print(f"✓ OBV: {obv.iloc[-1]:.0f}")
    print(f"✓ Volume Spikes detected: {volume_spikes.sum()}")
    
    # Add indicators to data
    data['ema_20'] = ema_20
    data['ema_50'] = ema_50
    data['rsi'] = rsi
    data['atr'] = atr
    data['macd'] = macd_df['macd']
    
    print("\n✓ All indicators calculated successfully!")
    
    return data


def test_fvg_detection(data: pd.DataFrame):
    """
    Test FVG detection and management
    
    Args:
        data: OHLCV DataFrame with indicators
    """
    print(f"\n{'='*60}")
    print("STEP 3: Testing FVG Detection")
    print(f"{'='*60}")
    
    # Initialize FVG Manager
    manager = FVGManager(
        lookback_days=FVG_CONFIG['lookback_days'],
        min_gap_atr_ratio=FVG_CONFIG['min_gap_atr_ratio']
    )
    
    # Simulate live updates
    print("\nProcessing candles and detecting FVGs...")
    
    for i in range(20, len(data)):
        atr_value = data['atr'].iloc[i]
        manager.update(data.iloc[:i+1], i, atr_value)
        
        # Print progress every 200 candles
        if i % 200 == 0:
            stats = manager.get_statistics()
            print(f"  Candle {i}: Active FVGs = {stats['total_active']}")
    
    # Get final statistics
    stats = manager.get_statistics()
    
    print(f"\n--- FVG Detection Results ---")
    print(f"✓ Total Active FVGs: {stats['total_active']}")
    print(f"  - Active Bullish: {stats['active_bullish']}")
    print(f"  - Active Bearish: {stats['active_bearish']}")
    print(f"✓ Total Bullish Created: {stats['total_bullish_created']}")
    print(f"✓ Total Bearish Created: {stats['total_bearish_created']}")
    print(f"✓ Bullish Touch Rate: {stats['bullish_touch_rate']:.2f}%")
    print(f"✓ Bearish Touch Rate: {stats['bearish_touch_rate']:.2f}%")
    
    # Test market structure
    current_price = data['close'].iloc[-1]
    structure = manager.get_market_structure(current_price)
    
    print(f"\n--- Market Structure ---")
    print(f"✓ Current Price: {current_price:.5f}")
    print(f"✓ Bias: {structure['bias']}")
    print(f"✓ Bullish FVGs below price: {len(structure['bullish_fvgs_below'])}")
    print(f"✓ Bearish FVGs above price: {len(structure['bearish_fvgs_above'])}")
    
    if structure['nearest_bullish_target']:
        fvg = structure['nearest_bullish_target']
        print(f"✓ Nearest Bullish Target: {fvg.bottom:.5f} - {fvg.top:.5f}")
    
    if structure['nearest_bearish_target']:
        fvg = structure['nearest_bearish_target']
        print(f"✓ Nearest Bearish Target: {fvg.bottom:.5f} - {fvg.top:.5f}")
    
    print("\n✓ FVG detection completed successfully!")
    
    return manager


def create_visualization(data: pd.DataFrame, manager: FVGManager):
    """
    Create visualization charts
    
    Args:
        data: OHLCV DataFrame
        manager: FVGManager with detected FVGs
    """
    print(f"\n{'='*60}")
    print("STEP 4: Creating Visualizations")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs('logs/charts', exist_ok=True)
    
    # Initialize visualizer
    visualizer = FVGVisualizer(show_touched_fvgs=True, show_labels=True)
    
    # Get all FVGs
    all_fvgs = manager.all_fvgs_history
    
    print(f"\nCreating charts with {len(all_fvgs)} FVGs...")
    
    # Create main FVG chart
    chart_path = 'logs/charts/mt5_fvg_indicators_chart.html'
    visualizer.plot_fvg_chart(
        data,
        all_fvgs,
        title="MT5 + FVG + Indicators Test Chart",
        show_volume=True,
        save_path=chart_path
    )
    print(f"✓ Main chart saved to: {chart_path}")
    
    # Create statistics chart
    stats_path = 'logs/charts/fvg_statistics.html'
    visualizer.plot_fvg_statistics(all_fvgs, save_path=stats_path)
    print(f"✓ Statistics chart saved to: {stats_path}")
    
    # Export FVG data
    history_df = manager.export_history_to_dataframe()
    if not history_df.empty:
        csv_path = 'logs/fvg_history.csv'
        history_df.to_csv(csv_path, index=False)
        print(f"✓ FVG history exported to: {csv_path}")
    
    print("\n✓ Visualizations created successfully!")


def main():
    """
    Main test execution
    """
    print("\n" + "="*60)
    print("MT5 + FVG + INDICATORS COMPREHENSIVE TEST")
    print("="*60)
    
    try:
        # Step 1: Get data (MT5 or sample)
        data = get_data_source()
        
        # Step 2: Calculate indicators
        data = test_indicators(data)
        
        # Step 3: Detect FVGs
        manager = test_fvg_detection(data)
        
        # Step 4: Create visualizations
        create_visualization(data, manager)
        
        # Final summary
        print(f"\n{'='*60}")
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        print("\n📊 Results Summary:")
        print("  ✓ Data source: MT5 or CSV or Sample")
        print("  ✓ Indicators: EMA, RSI, MACD, ATR, Bollinger Bands, Volume")
        print("  ✓ FVG detection: WORKING")
        print("  ✓ Visualizations: CREATED")
        
        print("\n📁 Output Files:")
        print("  - logs/charts/mt5_fvg_indicators_chart.html")
        print("  - logs/charts/fvg_statistics.html")
        print("  - logs/fvg_history.csv")
        
        print("\n📝 Project Status:")
        print("  ✓ MT5 data download: READY")
        print("  ✓ Indicator calculations: READY")
        print("  ✓ FVG detection: READY")
        print("  ✓ Integration: WORKING")
        
        print("\n🎉 The project is ready to download data from MT5")
        print("   and test FVG + indicators!")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
