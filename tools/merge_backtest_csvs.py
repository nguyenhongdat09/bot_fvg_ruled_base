"""
Merge Multiple Backtest CSVs

Gộp nhiều file backtest CSV từ các period khác nhau thành 1 file lớn
để phân tích statistical với nhiều data hơn.

Features:
1. Auto-detect tất cả backtest CSVs trong folder
2. Validate column consistency
3. Remove duplicates (based on entry_time)
4. Add source metadata
5. Sort by time
6. Output merged CSV

Usage:
    python tools/merge_backtest_csvs.py
    python tools/merge_backtest_csvs.py --folder data --output data/merged_backtest.csv

Author: Claude Code
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class BacktestCSVMerger:
    """
    Merge multiple backtest CSV files

    Usage:
        merger = BacktestCSVMerger('data')
        merged_df = merger.merge_all()
        merger.save('data/merged_backtest.csv')
    """

    def __init__(self, folder_path: str = 'data'):
        """
        Initialize merger

        Args:
            folder_path: Folder containing backtest CSV files
        """
        self.folder = Path(folder_path)
        self.csv_files = []
        self.merged_df = None

        print("=" * 80)
        print("BACKTEST CSV MERGER")
        print("=" * 80)

    def find_csv_files(self, pattern: str = 'backtest_*.csv'):
        """
        Find all backtest CSV files in folder

        Args:
            pattern: File pattern to match

        Returns:
            List of CSV file paths
        """
        print(f"\n[STEP 1] Finding CSV files in: {self.folder}")
        print(f"   Pattern: {pattern}")
        print("-" * 80)

        csv_files = list(self.folder.glob(pattern))

        if not csv_files:
            print(f"⚠️  No CSV files found matching pattern: {pattern}")
            return []

        # Sort by filename (typically has timestamp)
        csv_files = sorted(csv_files)

        print(f"✓ Found {len(csv_files)} CSV files:")
        for i, f in enumerate(csv_files, 1):
            try:
                # Quick peek to get row count
                df = pd.read_csv(f, nrows=1)
                total_rows = len(pd.read_csv(f))
                print(f"   {i}. {f.name:<50} ({total_rows:>4} trades)")
            except Exception as e:
                print(f"   {i}. {f.name:<50} (ERROR: {e})")

        self.csv_files = csv_files
        return csv_files

    def validate_columns(self):
        """
        Validate that all CSVs have compatible columns

        Returns:
            bool: True if all CSVs are compatible
        """
        print(f"\n[STEP 2] Validating Column Consistency")
        print("-" * 80)

        if not self.csv_files:
            print("⚠️  No CSV files to validate")
            return False

        # Load first file as reference
        ref_df = pd.read_csv(self.csv_files[0])
        ref_columns = set(ref_df.columns)

        print(f"✓ Reference columns ({len(ref_columns)}): {self.csv_files[0].name}")

        all_compatible = True
        incompatible_files = []

        for csv_file in self.csv_files[1:]:
            try:
                df = pd.read_csv(csv_file)
                columns = set(df.columns)

                # Check if columns match
                missing = ref_columns - columns
                extra = columns - ref_columns

                if missing or extra:
                    all_compatible = False
                    incompatible_files.append({
                        'file': csv_file.name,
                        'missing': missing,
                        'extra': extra
                    })

            except Exception as e:
                print(f"⚠️  Error reading {csv_file.name}: {e}")
                all_compatible = False

        if all_compatible:
            print(f"✓ All {len(self.csv_files)} CSV files have compatible columns!")
        else:
            print(f"\n⚠️  {len(incompatible_files)} files have incompatible columns:")
            for item in incompatible_files:
                print(f"\n   File: {item['file']}")
                if item['missing']:
                    print(f"   Missing columns: {item['missing']}")
                if item['extra']:
                    print(f"   Extra columns: {item['extra']}")

            print(f"\n⚠️  Will attempt to merge using common columns only")

        return all_compatible

    def merge_all(self, remove_duplicates: bool = True):
        """
        Merge all CSV files

        Args:
            remove_duplicates: Remove duplicate trades (by entry_time)

        Returns:
            Merged DataFrame
        """
        print(f"\n[STEP 3] Merging CSV Files")
        print("-" * 80)

        if not self.csv_files:
            print("⚠️  No CSV files to merge")
            return None

        dfs = []
        total_trades = 0

        for csv_file in self.csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Add source metadata
                df['source_file'] = csv_file.name

                dfs.append(df)
                total_trades += len(df)

                print(f"✓ Loaded {csv_file.name:<50} {len(df):>5} trades")

            except Exception as e:
                print(f"⚠️  Error loading {csv_file.name}: {e}")

        if not dfs:
            print("❌ No data loaded!")
            return None

        # Concatenate all dataframes
        print(f"\n✓ Concatenating {len(dfs)} dataframes...")
        merged_df = pd.concat(dfs, ignore_index=True)

        print(f"✓ Total trades before dedup: {len(merged_df)}")

        # Remove duplicates if requested
        if remove_duplicates and 'entry_time' in merged_df.columns:
            print(f"\n✓ Removing duplicates (based on entry_time)...")

            before_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=['entry_time'], keep='first')
            after_count = len(merged_df)

            duplicates_removed = before_count - after_count

            if duplicates_removed > 0:
                print(f"⚠️  Removed {duplicates_removed} duplicate trades")
            else:
                print(f"✓ No duplicates found")

            print(f"✓ Total trades after dedup: {after_count}")

        # Sort by entry_time
        if 'entry_time' in merged_df.columns:
            print(f"\n✓ Sorting by entry_time...")
            merged_df['entry_time'] = pd.to_datetime(merged_df['entry_time'])
            merged_df = merged_df.sort_values('entry_time').reset_index(drop=True)

            # Print date range
            min_date = merged_df['entry_time'].min()
            max_date = merged_df['entry_time'].max()
            print(f"✓ Date range: {min_date} to {max_date}")

        self.merged_df = merged_df

        return merged_df

    def print_summary(self):
        """Print summary statistics of merged data"""
        if self.merged_df is None:
            print("⚠️  No merged data available")
            return

        print(f"\n[SUMMARY] Merged Dataset Statistics")
        print("=" * 80)

        df = self.merged_df

        # Basic stats
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total Trades: {len(df)}")
        print(f"   Total Columns: {len(df.columns)}")

        # Win/Loss
        if 'pnl' in df.columns:
            wins = len(df[df['pnl'] > 0])
            losses = len(df[df['pnl'] <= 0])
            win_rate = wins / len(df) * 100

            total_pnl = df['pnl'].sum()
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
            avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losses > 0 else 0

            print(f"\n💰 PERFORMANCE:")
            print(f"   Wins: {wins} ({win_rate:.1f}%)")
            print(f"   Losses: {losses} ({100-win_rate:.1f}%)")
            print(f"   Total PnL: ${total_pnl:,.2f}")
            print(f"   Avg Win: ${avg_win:.2f}")
            print(f"   Avg Loss: ${avg_loss:.2f}")

            if losses > 0 and avg_loss != 0:
                profit_factor = abs(wins * avg_win / (losses * avg_loss))
                print(f"   Profit Factor: {profit_factor:.2f}")

        # Mode distribution
        if 'mode' in df.columns:
            print(f"\n📈 MODE DISTRIBUTION:")
            for mode, count in df['mode'].value_counts().items():
                pct = count / len(df) * 100
                print(f"   {mode}: {count} ({pct:.1f}%)")

        # Direction distribution
        if 'direction' in df.columns:
            print(f"\n↕️  DIRECTION DISTRIBUTION:")
            for direction, count in df['direction'].value_counts().items():
                pct = count / len(df) * 100
                print(f"   {direction}: {count} ({pct:.1f}%)")

        # Source files
        if 'source_file' in df.columns:
            print(f"\n📁 SOURCE FILES:")
            for source, count in df['source_file'].value_counts().items():
                pct = count / len(df) * 100
                print(f"   {source}: {count} ({pct:.1f}%)")

        # Date range
        if 'entry_time' in df.columns:
            print(f"\n📅 DATE RANGE:")
            print(f"   Start: {df['entry_time'].min()}")
            print(f"   End: {df['entry_time'].max()}")
            duration = (df['entry_time'].max() - df['entry_time'].min()).days
            print(f"   Duration: {duration} days")

        # Feature availability
        feature_cols = [
            'hurst', 'lr_deviation', 'r2', 'skewness', 'kurtosis',
            'obv_divergence', 'atr_percentile',
            'score_fvg', 'score_fvg_size_atr', 'score_hurst',
            'score_lr_deviation', 'score_skewness', 'score_kurtosis',
            'score_obv_div', 'score_regime'
        ]

        available_features = [f for f in feature_cols if f in df.columns]

        print(f"\n🔍 FEATURE AVAILABILITY:")
        print(f"   Available: {len(available_features)}/{len(feature_cols)}")
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            print(f"   Missing: {missing}")

    def save(self, output_path: str = 'data/merged_backtest.csv'):
        """
        Save merged data to CSV

        Args:
            output_path: Output file path
        """
        if self.merged_df is None:
            print("⚠️  No data to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.merged_df.to_csv(output_path, index=False)

        print(f"\n💾 Merged data saved to: {output_path}")
        print(f"   Total trades: {len(self.merged_df)}")
        print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    def run(self, pattern: str = 'backtest_*.csv',
            output_path: str = 'data/merged_backtest.csv'):
        """
        Run full merge pipeline

        Args:
            pattern: File pattern to match
            output_path: Output file path

        Returns:
            Merged DataFrame
        """
        try:
            # Find CSV files
            self.find_csv_files(pattern)

            if not self.csv_files:
                print("\n❌ No CSV files found!")
                return None

            # Validate columns
            self.validate_columns()

            # Merge all
            merged_df = self.merge_all(remove_duplicates=True)

            if merged_df is None:
                print("\n❌ Merge failed!")
                return None

            # Print summary
            self.print_summary()

            # Save
            self.save(output_path)

            print("\n" + "=" * 80)
            print("✅ MERGE COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            return merged_df

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Main function - Merge backtest CSVs

    Usage:
        python tools/merge_backtest_csvs.py
        python tools/merge_backtest_csvs.py --folder data --output data/merged.csv
    """
    import argparse

    parser = argparse.ArgumentParser(description='Merge Backtest CSVs')
    parser.add_argument('--folder', type=str, default='data',
                       help='Folder containing CSV files')
    parser.add_argument('--pattern', type=str, default='backtest_*.csv',
                       help='File pattern to match')
    parser.add_argument('--output', type=str, default='data/merged_backtest.csv',
                       help='Output file path')

    args = parser.parse_args()

    # Run merger
    merger = BacktestCSVMerger(args.folder)
    merged_df = merger.run(pattern=args.pattern, output_path=args.output)

    if merged_df is not None:
        print(f"\n🎉 SUCCESS! You can now use merged data for analysis:")
        print(f"\n   # Feature Optimization Pipeline:")
        print(f"   python tools/feature_optimization_pipeline.py --csv {args.output}")
        print(f"\n   # Losing Streak Analyzer:")
        print(f"   python tools/losing_streak_analyzer.py --csv {args.output}")

    return merger


if __name__ == '__main__':
    main()
