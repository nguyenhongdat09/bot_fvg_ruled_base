#!/usr/bin/env python3
"""
Download Real Market Data from MetaTrader 5

Script download du lieu that tu MT5
Su dung config.py de cau hinh MT5 path va settings
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from config import (
    MT5_CONFIG, DATA_CONFIG, DATA_DIR,
    get_mt5_path, get_data_filepath
)


class MT5DataDownloader:
    """
    MT5 Data Downloader
    Download historical data from MetaTrader 5
    """

    def __init__(self, mt5_path=None):
        """
        Initialize MT5 Data Downloader

        Args:
            mt5_path: Path to MT5 terminal (optional, use config if None)
        """
        self.mt5_path = mt5_path or get_mt5_path()
        self.connected = False

    def connect(self):
        """
        Connect to MT5

        Returns:
            bool: True if connected successfully
        """
        print("\n" + "="*60)
        print("CONNECTING TO METATRADER 5")
        print("="*60)

        # Initialize MT5 with path
        if self.mt5_path:
            print(f"\n= Using MT5 path: {self.mt5_path}")
            if not mt5.initialize(path=self.mt5_path):
                print(f"L Failed to initialize MT5 with path: {self.mt5_path}")
                print(f"   Error: {mt5.last_error()}")
                return False
        else:
            print("\n= Using default MT5 installation")
            if not mt5.initialize():
                print(f"L Failed to initialize MT5")
                print(f"   Error: {mt5.last_error()}")
                print("\n= Troubleshooting:")
                print("   1. Make sure MetaTrader 5 is installed")
                print("   2. Check MT5 path in config.py")
                print("   3. Try opening MT5 manually first")
                return False

        self.connected = True
        print(" Connected to MT5")

        # Print MT5 info
        version = mt5.version()
        print(f"\n= MT5 Info:")
        print(f"   Version: {version[0]}.{version[1]}")
        print(f"   Build: {version[2]}")

        # Account info
        account_info = mt5.account_info()
        if account_info:
            print(f"\n=d Account:")
            print(f"   Login: {account_info.login}")
            print(f"   Server: {account_info.server}")
            print(f"   Company: {account_info.company}")
            print(f"   Balance: ${account_info.balance:.2f}")
        else:
            print("\n  No account logged in")
            print("   You can still download data from some brokers")

        return True

    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("\n MT5 connection closed")

    def download_data(
        self,
        symbol='EURUSD',
        timeframe_name='M15',
        days=30,
        start_date=None,
        end_date=None,
        save_csv=True,
        output_path=None
    ):
        """
        Download historical data

        Args:
            symbol: Trading symbol (EURUSD, GBPUSD, etc.)
            timeframe_name: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            days: Number of days to download
            start_date: Start date (datetime or str 'YYYY-MM-DD')
            end_date: End date (datetime or str 'YYYY-MM-DD')
            save_csv: Save to CSV file
            output_path: Custom output path (optional)

        Returns:
            pd.DataFrame: Downloaded data or None if failed
        """
        if not self.connected:
            print("L Not connected to MT5. Call connect() first.")
            return None

        print("\n" + "="*60)
        print(f"DOWNLOADING: {symbol} {timeframe_name}")
        print("="*60)

        # 1. Check and enable symbol
        print(f"\n1. Checking symbol {symbol}...")
        symbol_info = mt5.symbol_info(symbol)

        if symbol_info is None:
            print(f"L Symbol {symbol} not found")

            # Show available symbols
            print("\n= Available symbols:")
            available = mt5.symbols_get()
            if available:
                # Group by first 3 letters
                symbols_dict = {}
                for s in available:
                    prefix = s.name[:3]
                    if prefix not in symbols_dict:
                        symbols_dict[prefix] = []
                    symbols_dict[prefix].append(s.name)

                # Show first few groups
                for prefix in sorted(symbols_dict.keys())[:5]:
                    symbols = symbols_dict[prefix][:5]
                    print(f"   {prefix}*: {', '.join(symbols)}")

            return None

        # Enable symbol if not visible
        if not symbol_info.visible:
            print(f"   Enabling {symbol}...")
            if not mt5.symbol_select(symbol, True):
                print(f"L Failed to enable {symbol}")
                return None

        print(f" Symbol {symbol} ready")
        print(f"   Bid: {symbol_info.bid:.5f}")
        print(f"   Ask: {symbol_info.ask:.5f}")
        print(f"   Spread: {symbol_info.spread} points")
        print(f"   Digits: {symbol_info.digits}")

        # 2. Set timeframe
        print(f"\n2. Setting timeframe {timeframe_name}...")
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }

        if timeframe_name not in timeframe_map:
            print(f"L Invalid timeframe: {timeframe_name}")
            print(f"   Available: {list(timeframe_map.keys())}")
            return None

        timeframe = timeframe_map[timeframe_name]
        print(f" Timeframe: {timeframe_name}")

        # 3. Set date range
        print(f"\n3. Setting date range...")

        if start_date is None and end_date is None:
            # Use days parameter
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        else:
            # Convert string to datetime if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')

        print(f"   From: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   To:   {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Duration: {(end_date - start_date).days} days")

        # 4. Download data
        print(f"\n4. Downloading data...")
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            print(f"L No data received")
            print(f"   Error code: {error[0]}")
            print(f"   Error message: {error[1]}")
            print("\n= Possible reasons:")
            print("   - Symbol not available for this timeframe")
            print("   - Date range too old (no historical data)")
            print("   - MT5 not logged in")
            return None

        print(f" Downloaded {len(rates)} candles")

        # 5. Convert to DataFrame
        print(f"\n5. Converting to DataFrame...")
        data = pd.DataFrame(rates)

        # Convert timestamp
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data.set_index('time', inplace=True)

        # Rename columns
        data.rename(columns={
            'tick_volume': 'volume',
        }, inplace=True)

        # Keep only OHLCV
        data = data[['open', 'high', 'low', 'close', 'volume']]

        print(f" Data converted")
        print(f"\n= Data Summary:")
        print(f"   Rows: {len(data)}")
        print(f"   Columns: {list(data.columns)}")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Price range: {data['low'].min():.5f} to {data['high'].max():.5f}")
        print(f"   Avg volume: {data['volume'].mean():.0f}")

        print(f"\n   First 3 candles:")
        print(data.head(3))

        # 6. Save to CSV
        if save_csv:
            print(f"\n6. Saving to CSV...")

            if output_path is None:
                output_path = get_data_filepath(symbol, timeframe_name, days)
            else:
                # Convert to Path object if string
                output_path = Path(output_path)

            # Create directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            data.to_csv(output_path)

            file_size = os.path.getsize(output_path) / 1024
            print(f" Saved to: {output_path}")
            print(f"   File size: {file_size:.1f} KB")

        return data


def main():
    """Main function"""

    print("\n" + "="*70)
    print("  MT5 DATA DOWNLOADER")
    print("="*70)

    # Show current config
    print("\n= Current Configuration:")
    print(f"   MT5 Path: {MT5_CONFIG['path'] or 'Default'}")
    print(f"   Symbol: {DATA_CONFIG['symbol']}")
    print(f"   Timeframe: {DATA_CONFIG['timeframe']}")
    print(f"   Days: {DATA_CONFIG['days']}")

    # Initialize downloader
    downloader = MT5DataDownloader()

    # Connect to MT5
    if not downloader.connect():
        print("\nL Connection failed!")
        print("\n= To fix:")
        print("   1. Edit config.py")
        print("   2. Set MT5_CONFIG['path'] to your MT5 terminal path")
        print("      Example: r'C:\\Program Files\\MetaTrader 5\\terminal64.exe'")
        return 1

    try:
        # Download data
        data = downloader.download_data(
            symbol=DATA_CONFIG['symbol'],
            timeframe_name=DATA_CONFIG['timeframe'],
            days=DATA_CONFIG['days'],
            save_csv=True
        )

        if data is None:
            print("\nL Download failed!")
            return 1

        print("\n" + "="*70)
        print("  DOWNLOAD COMPLETED ")
        print("="*70)

        filepath = get_data_filepath()
        print(f"\n= Data saved to: {filepath}")
        print(f"\n= Next steps:")
        print(f"   1. Check file: {filepath}")
        print(f"   2. Test with real data: python test_fvg_real_data.py")
        print(f"   3. Or use in your own script:")
        print(f"      >>> import pandas as pd")
        print(f"      >>> data = pd.read_csv('{filepath}', index_col='time', parse_dates=True)")

    finally:
        # Disconnect
        downloader.disconnect()

    return 0


if __name__ == '__main__':
    exit(main())
