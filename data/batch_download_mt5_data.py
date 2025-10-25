#!/usr/bin/env python3
"""
BATCH DOWNLOAD MT5 DATA

Script download NHIEU cap tien + NHIEU timeframes cung luc.
Chi can chay 1 lan!

Usage:
    python data/batch_download_mt5_data.py

Config:
    Edit config.py -> BATCH_DOWNLOAD_CONFIG
    - symbols: List of symbols to download
    - timeframes: List of timeframes for each symbol
    - days: Number of days to download

Author: Claude Code
Date: 2025-10-24
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from download_mt5_data import MT5DataDownloader
from config import BATCH_DOWNLOAD_CONFIG, DATA_DIR


class BatchDownloader:
    """Batch downloader for multiple symbols and timeframes"""

    def __init__(self, config=None):
        """
        Initialize batch downloader

        Args:
            config: Batch download config (default: BATCH_DOWNLOAD_CONFIG)
        """
        self.config = config or BATCH_DOWNLOAD_CONFIG
        self.downloader = MT5DataDownloader()
        self.results = {
            'success': [],
            'skipped': [],
            'failed': [],
        }

    def run(self):
        """Run batch download"""

        print("\n" + "="*80)
        print("BATCH DOWNLOAD MT5 DATA")
        print("="*80)

        if not self.config['enabled']:
            print("\n⚠️  Batch download is DISABLED in config")
            print("   Enable it: config.BATCH_DOWNLOAD_CONFIG['enabled'] = True")
            return

        symbols = self.config['symbols']
        timeframes = self.config['timeframes']
        days = self.config['days']

        print(f"\n📊 Configuration:")
        print(f"   Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
        print(f"   Timeframes: {len(timeframes)} ({', '.join(timeframes)})")
        print(f"   Days: {days}")
        print(f"   Total downloads: {len(symbols) * len(timeframes)}")
        print(f"   Skip existing: {self.config['skip_existing']}")

        # Connect to MT5
        print("\n" + "="*80)
        if not self.downloader.connect():
            print("\n❌ Failed to connect to MT5")
            return

        # Download each symbol + timeframe combination
        print("\n" + "="*80)
        print("DOWNLOADING...")
        print("="*80)

        total = len(symbols) * len(timeframes)
        current = 0

        for symbol in symbols:
            for timeframe in timeframes:
                current += 1

                # Progress
                if self.config['show_progress']:
                    print(f"\n[{current}/{total}] {symbol} {timeframe}")
                    print("-" * 60)

                # Check if file exists
                filename = f"{symbol}_{timeframe}_{days}days.csv"
                filepath = DATA_DIR / filename

                if filepath.exists() and self.config['skip_existing']:
                    print(f"⏭️  Skipping (file exists): {filename}")
                    self.results['skipped'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'file': filename
                    })
                    continue

                # Download with retries
                success = self._download_with_retry(
                    symbol, timeframe, days, filepath
                )

                if success:
                    self.results['success'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'file': filename
                    })
                else:
                    self.results['failed'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'file': filename
                    })

                    if not self.config['continue_on_error']:
                        print("\n❌ Stopping due to error (continue_on_error=False)")
                        break

                # Delay between downloads
                if current < total:
                    time.sleep(self.config['delay_between'])

            # Break outer loop if needed
            if not self.config['continue_on_error'] and self.results['failed']:
                break

        # Disconnect
        self.downloader.disconnect()

        # Print summary
        self._print_summary()

    def _download_with_retry(self, symbol, timeframe, days, filepath):
        """
        Download with retry logic

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            days: Number of days
            filepath: Output file path

        Returns:
            bool: True if successful
        """
        max_retries = self.config['max_retries']

        for attempt in range(max_retries + 1):
            try:
                # Download
                data = self.downloader.download_data(
                    symbol=symbol,
                    timeframe_name=timeframe,
                    days=days,
                    save_csv=True,
                    output_path=str(filepath)
                )

                if data is not None and len(data) > 0:
                    print(f"✅ Success: {len(data)} candles -> {filepath.name}")
                    return True

                # Failed but no exception
                if attempt < max_retries:
                    print(f"⚠️  Retry {attempt + 1}/{max_retries}...")
                    time.sleep(1)
                else:
                    print(f"❌ Failed after {max_retries} retries")
                    return False

            except Exception as e:
                print(f"❌ Error: {e}")

                if attempt < max_retries:
                    print(f"⚠️  Retry {attempt + 1}/{max_retries}...")
                    time.sleep(1)
                else:
                    print(f"❌ Failed after {max_retries} retries")
                    return False

        return False

    def _print_summary(self):
        """Print download summary"""
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)

        print(f"\n✅ Successful: {len(self.results['success'])}")
        if self.results['success']:
            for item in self.results['success'][:10]:  # Show first 10
                print(f"   {item['symbol']:8s} {item['timeframe']:4s} -> {item['file']}")
            if len(self.results['success']) > 10:
                print(f"   ... and {len(self.results['success']) - 10} more")

        print(f"\n⏭️  Skipped: {len(self.results['skipped'])}")
        if self.results['skipped'] and len(self.results['skipped']) <= 10:
            for item in self.results['skipped']:
                print(f"   {item['symbol']:8s} {item['timeframe']:4s} -> {item['file']}")

        print(f"\n❌ Failed: {len(self.results['failed'])}")
        if self.results['failed']:
            for item in self.results['failed']:
                print(f"   {item['symbol']:8s} {item['timeframe']:4s} -> {item['file']}")

        print(f"\n📁 Files saved to: {DATA_DIR}")
        print("="*80)

        # Overall status
        total_attempted = len(self.results['success']) + len(self.results['failed'])
        if total_attempted > 0:
            success_rate = len(self.results['success']) / total_attempted * 100
            print(f"\n📊 Success Rate: {success_rate:.1f}% ({len(self.results['success'])}/{total_attempted})")


def main():
    """Main function"""
    batch = BatchDownloader()
    batch.run()


if __name__ == '__main__':
    main()
