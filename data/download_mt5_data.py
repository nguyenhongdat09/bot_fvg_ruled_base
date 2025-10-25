"""
MT5 Data Downloader - Download historical data from MetaTrader 5
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MT5_CONFIG, OUTPUT_CONFIG


def initialize_mt5(login: Optional[int] = None, 
                   password: Optional[str] = None, 
                   server: Optional[str] = None) -> bool:
    """
    Initialize connection to MT5
    
    Args:
        login: MT5 login ID (uses config if None)
        password: MT5 password (uses config if None)
        server: MT5 server (uses config if None)
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Use config values if not provided
    login = login or MT5_CONFIG.get('login')
    password = password or MT5_CONFIG.get('password')
    server = server or MT5_CONFIG.get('server')
    
    # Initialize MT5
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    
    # If credentials provided, login
    if login and password and server:
        if not mt5.login(login, password, server):
            print(f"❌ MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        print(f"✓ Connected to MT5 - Account: {login}, Server: {server}")
    else:
        print(f"✓ MT5 initialized (no login credentials provided)")
    
    return True


def shutdown_mt5():
    """Shutdown MT5 connection"""
    mt5.shutdown()
    print("✓ MT5 connection closed")


def get_mt5_timeframe(timeframe_str: str) -> int:
    """
    Convert timeframe string to MT5 timeframe constant
    
    Args:
        timeframe_str: Timeframe string (M1, M5, M15, M30, H1, H4, D1)
    
    Returns:
        int: MT5 timeframe constant
    """
    
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }
    
    return timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_M15)


def download_ohlcv_data(symbol: str,
                        timeframe: str = 'M15',
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        num_bars: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Download OHLCV data from MT5
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1)
        start_date: Start date (if None, calculated from num_bars)
        end_date: End date (if None, uses current time)
        num_bars: Number of bars to download (if start_date is None)
    
    Returns:
        pd.DataFrame: OHLCV data with columns [open, high, low, close, volume]
                     Index is datetime
                     Returns None if download fails
    """
    
    # Get MT5 timeframe
    mt5_timeframe = get_mt5_timeframe(timeframe)
    
    # Set end date to now if not provided
    if end_date is None:
        end_date = datetime.now()
    
    # Download data
    if start_date is not None:
        # Download by date range
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
    elif num_bars is not None:
        # Download by number of bars
        rates = mt5.copy_rates_from(symbol, mt5_timeframe, end_date, num_bars)
    else:
        # Default: download last 1000 bars
        rates = mt5.copy_rates_from(symbol, mt5_timeframe, end_date, 1000)
    
    # Check if download was successful
    if rates is None or len(rates) == 0:
        print(f"❌ Failed to download data for {symbol}: {mt5.last_error()}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    # Convert time to datetime and set as index
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Rename columns to standard names
    df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume'
    }, inplace=True)
    
    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]
    
    print(f"✓ Downloaded {len(df)} bars for {symbol} ({timeframe})")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def save_data_to_csv(df: pd.DataFrame, 
                     symbol: str, 
                     timeframe: str,
                     output_dir: Optional[str] = None) -> str:
    """
    Save OHLCV data to CSV file
    
    Args:
        df: OHLCV DataFrame
        symbol: Trading symbol
        timeframe: Timeframe string
        output_dir: Output directory (uses config if None)
    
    Returns:
        str: Path to saved CSV file
    """
    
    # Use config output dir if not provided
    if output_dir is None:
        output_dir = OUTPUT_CONFIG['data_dir']
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"{symbol}_{timeframe}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save to CSV
    df.to_csv(filepath)
    
    print(f"✓ Data saved to: {filepath}")
    
    return filepath


def load_data_from_csv(symbol: str,
                       timeframe: str,
                       data_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from CSV file
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        data_dir: Data directory (uses config if None)
    
    Returns:
        pd.DataFrame: OHLCV data or None if file doesn't exist
    """
    
    # Use config data dir if not provided
    if data_dir is None:
        data_dir = OUTPUT_CONFIG['data_dir']
    
    # Generate filename
    filename = f"{symbol}_{timeframe}.csv"
    filepath = os.path.join(data_dir, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    # Load from CSV
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    print(f"✓ Data loaded from: {filepath}")
    print(f"  {len(df)} bars, {df.index[0]} to {df.index[-1]}")
    
    return df


def download_multiple_symbols(symbols: list,
                              timeframe: str = 'M15',
                              num_bars: int = 1000,
                              save_csv: bool = True) -> dict:
    """
    Download data for multiple symbols
    
    Args:
        symbols: List of symbol strings
        timeframe: Timeframe string
        num_bars: Number of bars to download
        save_csv: Save to CSV files
    
    Returns:
        dict: {symbol: DataFrame} mapping
    """
    
    data_dict = {}
    
    print(f"\n{'='*60}")
    print(f"Downloading data for {len(symbols)} symbols...")
    print(f"{'='*60}\n")
    
    for symbol in symbols:
        df = download_ohlcv_data(symbol, timeframe, num_bars=num_bars)
        
        if df is not None:
            data_dict[symbol] = df
            
            if save_csv:
                save_data_to_csv(df, symbol, timeframe)
        
        print()  # Empty line between symbols
    
    print(f"✓ Downloaded data for {len(data_dict)}/{len(symbols)} symbols")
    
    return data_dict


# ===== MAIN EXECUTION =====

def main():
    """
    Main execution - Download sample data
    """
    
    print("\n" + "="*60)
    print("MT5 DATA DOWNLOADER")
    print("="*60)
    
    # Initialize MT5
    if not initialize_mt5():
        return 1
    
    try:
        # Example 1: Download single symbol
        print("\n--- Example 1: Download EURUSD M15 data ---")
        df_eurusd = download_ohlcv_data('EURUSD', 'M15', num_bars=1000)
        
        if df_eurusd is not None:
            save_data_to_csv(df_eurusd, 'EURUSD', 'M15')
            print(f"\nData preview:")
            print(df_eurusd.head())
        
        # Example 2: Download multiple symbols
        print("\n--- Example 2: Download multiple symbols ---")
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        data_dict = download_multiple_symbols(symbols, 'M15', num_bars=500)
        
        # Example 3: Download by date range
        print("\n--- Example 3: Download by date range ---")
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)
        df_range = download_ohlcv_data('EURUSD', 'H1', start_date=start_date, end_date=end_date)
        
        if df_range is not None:
            print(f"Downloaded {len(df_range)} bars from {start_date} to {end_date}")
        
        print("\n" + "="*60)
        print("✓ DOWNLOAD COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Shutdown MT5
        shutdown_mt5()
    
    return 0


if __name__ == '__main__':
    exit(main())
