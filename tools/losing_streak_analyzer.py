"""
Losing Streak Analyzer

Phân tích chuỗi thua liên tiếp để tìm patterns và nguyên nhân gốc rễ.

Mục đích:
1. Identify tất cả losing streaks >= threshold
2. Extract features của trades trong streaks
3. Compare với winning streaks và random trades
4. Tìm common patterns dẫn đến long losing streaks
5. Đề xuất filters để giảm losing streak length

Author: Claude Code
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy import stats


class LosingStreakAnalyzer:
    """
    Phân tích chuỗi thua để tìm patterns

    Usage:
        analyzer = LosingStreakAnalyzer('data/backtest.csv')
        results = analyzer.analyze(min_streak_length=6)
        analyzer.save_report('output/losing_streak_report.csv')
    """

    def __init__(self, csv_path: str):
        """
        Initialize analyzer

        Args:
            csv_path: Path to backtest CSV
        """
        self.csv_path = Path(csv_path)
        self.df = None
        self.streaks = []
        self.results = {}

        print("=" * 80)
        print("LOSING STREAK ANALYZER")
        print("=" * 80)

    def load_data(self):
        """Load backtest data"""
        print(f"\n[STEP 1] Loading data from: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(self.df)} trades")

        # Add win/loss flag
        self.df['is_win'] = (self.df['pnl'] > 0).astype(int)

        return self.df

    def identify_streaks(self, min_length: int = 1):
        """
        Identify all losing and winning streaks

        Args:
            min_length: Minimum streak length to record

        Returns:
            List of streak dictionaries
        """
        print(f"\n[STEP 2] Identifying Streaks (min_length={min_length})")
        print("-" * 80)

        streaks = []
        current_streak = []
        current_type = None

        for idx, row in self.df.iterrows():
            is_win = row['is_win']

            # Start new streak or continue
            if current_type is None:
                current_type = 'WIN' if is_win else 'LOSS'
                current_streak = [idx]
            elif (is_win and current_type == 'WIN') or (not is_win and current_type == 'LOSS'):
                # Continue streak
                current_streak.append(idx)
            else:
                # Streak ended, record it
                if len(current_streak) >= min_length:
                    streaks.append({
                        'type': current_type,
                        'length': len(current_streak),
                        'start_idx': current_streak[0],
                        'end_idx': current_streak[-1],
                        'trade_indices': current_streak
                    })

                # Start new streak
                current_type = 'WIN' if is_win else 'LOSS'
                current_streak = [idx]

        # Don't forget last streak
        if len(current_streak) >= min_length:
            streaks.append({
                'type': current_type,
                'length': len(current_streak),
                'start_idx': current_streak[0],
                'end_idx': current_streak[-1],
                'trade_indices': current_streak
            })

        self.streaks = streaks

        # Statistics
        loss_streaks = [s for s in streaks if s['type'] == 'LOSS']
        win_streaks = [s for s in streaks if s['type'] == 'WIN']

        print(f"✓ Total Streaks: {len(streaks)}")
        print(f"   • Losing Streaks: {len(loss_streaks)}")
        print(f"   • Winning Streaks: {len(win_streaks)}")

        if loss_streaks:
            loss_lengths = [s['length'] for s in loss_streaks]
            print(f"\n✓ Losing Streak Statistics:")
            print(f"   • Max Length: {max(loss_lengths)}")
            print(f"   • Avg Length: {np.mean(loss_lengths):.1f}")
            print(f"   • Median Length: {np.median(loss_lengths):.1f}")
            print(f"   • Streaks >= 6: {len([l for l in loss_lengths if l >= 6])}")
            print(f"   • Streaks >= 10: {len([l for l in loss_lengths if l >= 10])}")

        return streaks

    def analyze_long_losing_streaks(self, min_length: int = 6):
        """
        Phân tích chi tiết các losing streaks dài

        Args:
            min_length: Minimum length to consider "long"

        Returns:
            DataFrame with analysis results
        """
        print(f"\n[STEP 3] Analyzing Long Losing Streaks (>= {min_length})")
        print("=" * 80)

        # Get long losing streaks
        long_loss_streaks = [
            s for s in self.streaks
            if s['type'] == 'LOSS' and s['length'] >= min_length
        ]

        if not long_loss_streaks:
            print(f"⚠️  No losing streaks >= {min_length} found!")
            return pd.DataFrame()

        print(f"✓ Found {len(long_loss_streaks)} long losing streaks\n")

        # Analyze each streak
        streak_analyses = []

        for i, streak in enumerate(long_loss_streaks, 1):
            print(f"[Streak #{i}] Length: {streak['length']} trades")
            print(f"   Trades: #{streak['start_idx']+1} to #{streak['end_idx']+1}")

            # Extract trades in streak
            streak_trades = self.df.iloc[streak['trade_indices']]

            # Calculate statistics
            analysis = {
                'streak_id': i,
                'length': streak['length'],
                'start_trade': streak['start_idx'] + 1,
                'end_trade': streak['end_idx'] + 1,
                'total_loss': streak_trades['pnl'].sum(),

                # Feature averages
                'avg_confluence_score': streak_trades['confluence_score'].mean(),
                'avg_hurst': streak_trades.get('hurst', pd.Series([0])).mean(),
                'avg_lr_deviation': streak_trades.get('lr_deviation', pd.Series([0])).mean(),
                'avg_r2': streak_trades.get('r2', pd.Series([0])).mean(),
                'avg_skewness': streak_trades.get('skewness', pd.Series([0])).mean(),
                'avg_kurtosis': streak_trades.get('kurtosis', pd.Series([0])).mean(),
                'avg_atr_percentile': streak_trades.get('atr_percentile', pd.Series([0])).mean(),

                # Mode distribution
                'virtual_trades': len(streak_trades[streak_trades['mode'] == 'VIRTUAL']),
                'real_trades': len(streak_trades[streak_trades['mode'] == 'REAL']),

                # Direction distribution
                'buy_trades': len(streak_trades[streak_trades['direction'] == 'BUY']),
                'sell_trades': len(streak_trades[streak_trades['direction'] == 'SELL']),
            }

            streak_analyses.append(analysis)

            print(f"   Total Loss: ${analysis['total_loss']:.2f}")
            print(f"   Avg Confluence: {analysis['avg_confluence_score']:.1f}")
            print(f"   Direction: {analysis['buy_trades']} BUY, {analysis['sell_trades']} SELL")
            print()

        streak_df = pd.DataFrame(streak_analyses)
        self.results['long_streaks'] = streak_df

        return streak_df

    def compare_with_winners(self):
        """
        So sánh features của losing streaks vs winning streaks
        """
        print(f"\n[STEP 4] Comparing Losing Streaks vs Winning Streaks")
        print("=" * 80)

        # Get long losing streaks (>= 6)
        long_loss_streaks = [
            s for s in self.streaks
            if s['type'] == 'LOSS' and s['length'] >= 6
        ]

        # Get winning streaks (>= 3 for comparison)
        win_streaks = [
            s for s in self.streaks
            if s['type'] == 'WIN' and s['length'] >= 3
        ]

        if not long_loss_streaks or not win_streaks:
            print("⚠️  Not enough data for comparison")
            return None

        # Extract all trades from losing streaks
        loss_streak_indices = []
        for streak in long_loss_streaks:
            loss_streak_indices.extend(streak['trade_indices'])

        loss_streak_trades = self.df.iloc[loss_streak_indices]

        # Extract all trades from winning streaks
        win_streak_indices = []
        for streak in win_streaks:
            win_streak_indices.extend(streak['trade_indices'])

        win_streak_trades = self.df.iloc[win_streak_indices]

        # Compare features
        features_to_compare = [
            'confluence_score',
            'hurst',
            'lr_deviation',
            'r2',
            'skewness',
            'kurtosis',
            'atr_percentile'
        ]

        comparison = []

        for feature in features_to_compare:
            if feature not in self.df.columns:
                continue

            loss_values = loss_streak_trades[feature].dropna()
            win_values = win_streak_trades[feature].dropna()

            if len(loss_values) == 0 or len(win_values) == 0:
                continue

            # Statistical test
            try:
                t_stat, p_value = stats.ttest_ind(loss_values, win_values)
            except:
                t_stat, p_value = 0, 1

            comparison.append({
                'feature': feature,
                'loss_streak_mean': loss_values.mean(),
                'win_streak_mean': win_values.mean(),
                'difference': loss_values.mean() - win_values.mean(),
                'loss_streak_std': loss_values.std(),
                'win_streak_std': win_values.std(),
                'p_value': p_value,
                'significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
            })

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('p_value')

        print("\n✓ FEATURE COMPARISON (Losing Streaks vs Winning Streaks):")
        print("-" * 80)
        print(f"{'Feature':<20} {'Loss Streak':<15} {'Win Streak':<15} {'Difference':<12} {'p-value':<10}")
        print("-" * 80)

        for _, row in comparison_df.iterrows():
            print(f"{row['feature']:<20} "
                  f"{row['loss_streak_mean']:>14.4f} "
                  f"{row['win_streak_mean']:>14.4f} "
                  f"{row['difference']:>11.4f} "
                  f"{row['p_value']:>9.4f} {row['significant']}")

        print("\n* p<0.05  ** p<0.01  *** p<0.001 (significant difference)")

        self.results['comparison'] = comparison_df

        return comparison_df

    def identify_common_patterns(self):
        """
        Tìm patterns chung trong losing streaks
        """
        print(f"\n[STEP 5] Identifying Common Patterns")
        print("=" * 80)

        # Get long losing streaks
        long_loss_streaks = [
            s for s in self.streaks
            if s['type'] == 'LOSS' and s['length'] >= 6
        ]

        if not long_loss_streaks:
            print("⚠️  No long losing streaks to analyze")
            return

        # Extract all trades
        all_loss_indices = []
        for streak in long_loss_streaks:
            all_loss_indices.extend(streak['trade_indices'])

        loss_trades = self.df.iloc[all_loss_indices]

        patterns = []

        # Pattern 1: Low confidence scores
        low_conf_pct = len(loss_trades[loss_trades['confluence_score'] < 75]) / len(loss_trades) * 100
        patterns.append({
            'pattern': 'Low Confluence Score (<75)',
            'percentage': low_conf_pct,
            'description': f'{low_conf_pct:.1f}% of losing streak trades had score < 75'
        })

        # Pattern 2: Market regime
        if 'atr_percentile' in loss_trades.columns:
            high_vol = len(loss_trades[loss_trades['atr_percentile'] > 70]) / len(loss_trades) * 100
            low_vol = len(loss_trades[loss_trades['atr_percentile'] < 30]) / len(loss_trades) * 100

            patterns.append({
                'pattern': 'High Volatility Regime (ATR > 70%ile)',
                'percentage': high_vol,
                'description': f'{high_vol:.1f}% in high volatility'
            })

            patterns.append({
                'pattern': 'Low Volatility Regime (ATR < 30%ile)',
                'percentage': low_vol,
                'description': f'{low_vol:.1f}% in low volatility'
            })

        # Pattern 3: Hurst values
        if 'hurst' in loss_trades.columns:
            mean_rev = len(loss_trades[loss_trades['hurst'] < 0.45]) / len(loss_trades) * 100

            patterns.append({
                'pattern': 'Mean Reverting Market (Hurst < 0.45)',
                'percentage': mean_rev,
                'description': f'{mean_rev:.1f}% traded in mean-reverting conditions'
            })

        # Pattern 4: Direction bias
        buy_pct = len(loss_trades[loss_trades['direction'] == 'BUY']) / len(loss_trades) * 100
        sell_pct = len(loss_trades[loss_trades['direction'] == 'SELL']) / len(loss_trades) * 100

        patterns.append({
            'pattern': 'Direction Bias',
            'percentage': max(buy_pct, sell_pct),
            'description': f'BUY: {buy_pct:.1f}%, SELL: {sell_pct:.1f}%'
        })

        # Print patterns
        print("\n✓ COMMON PATTERNS IN LOSING STREAKS:")
        print("-" * 80)

        for p in patterns:
            print(f"• {p['pattern']:<40} {p['percentage']:>6.1f}%")
            print(f"  → {p['description']}")

        self.results['patterns'] = patterns

        return patterns

    def generate_recommendations(self):
        """
        Generate actionable recommendations để giảm losing streaks
        """
        print(f"\n[STEP 6] Generating Recommendations")
        print("=" * 80)

        recommendations = []

        # Based on comparison results
        if 'comparison' in self.results:
            comp = self.results['comparison']

            # Significant differences
            sig_features = comp[comp['p_value'] < 0.05]

            for _, row in sig_features.iterrows():
                feature = row['feature']
                diff = row['difference']

                if diff > 0:
                    # Losing streaks have HIGHER values
                    rec = f"FILTER: Avoid trades when {feature} > {row['loss_streak_mean']:.2f}"
                    impact = "Potential to reduce losing streak length"
                else:
                    # Losing streaks have LOWER values
                    rec = f"FILTER: Avoid trades when {feature} < {row['loss_streak_mean']:.2f}"
                    impact = "Potential to reduce losing streak length"

                recommendations.append({
                    'recommendation': rec,
                    'impact': impact,
                    'p_value': row['p_value']
                })

        # Based on patterns
        if 'patterns' in self.results:
            patterns = self.results['patterns']

            for p in patterns:
                if p['percentage'] > 60:  # If > 60% of losing streaks have this pattern
                    if 'High Volatility' in p['pattern']:
                        recommendations.append({
                            'recommendation': 'ADD FILTER: Skip trades when ATR percentile > 70',
                            'impact': f"Could avoid {p['percentage']:.0f}% of losing streak trades",
                            'p_value': 0.0
                        })

                    if 'Mean Reverting' in p['pattern']:
                        recommendations.append({
                            'recommendation': 'ADD FILTER: Skip trades when Hurst < 0.45',
                            'impact': f"Could avoid {p['percentage']:.0f}% of losing streak trades",
                            'p_value': 0.0
                        })

                    if 'Low Confluence' in p['pattern']:
                        recommendations.append({
                            'recommendation': 'INCREASE min_confidence_score from 70 to 75',
                            'impact': f"Could filter out {p['percentage']:.0f}% of losing streak trades",
                            'p_value': 0.0
                        })

        # Print recommendations
        print("\n✅ ACTIONABLE RECOMMENDATIONS:")
        print("-" * 80)

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['recommendation']}")
                print(f"   Impact: {rec['impact']}")
                if rec['p_value'] > 0:
                    print(f"   Confidence: p={rec['p_value']:.4f}")
        else:
            print("⚠️  Not enough statistical evidence for specific recommendations")
            print("\nGeneral recommendations:")
            print("1. Increase min_confidence_score to filter marginal trades")
            print("2. Add market regime filter (avoid extreme volatility)")
            print("3. Consider reducing position size during unfavorable conditions")

        self.results['recommendations'] = recommendations

        return recommendations

    def analyze(self, min_streak_length: int = 6):
        """
        Run full analysis pipeline

        Args:
            min_streak_length: Minimum losing streak length to analyze

        Returns:
            Dictionary with all analysis results
        """
        try:
            # Load data
            self.load_data()

            # Identify streaks
            self.identify_streaks(min_length=1)

            # Analyze long losing streaks
            self.analyze_long_losing_streaks(min_streak_length)

            # Compare with winners
            self.compare_with_winners()

            # Identify patterns
            self.identify_common_patterns()

            # Generate recommendations
            self.generate_recommendations()

            print("\n" + "=" * 80)
            print("✅ ANALYSIS COMPLETED!")
            print("=" * 80)

            return self.results

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_report(self, output_path: str = 'output/losing_streak_report.csv'):
        """Save detailed report to CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if 'long_streaks' in self.results:
            self.results['long_streaks'].to_csv(output_path, index=False)
            print(f"\n💾 Detailed report saved to: {output_path}")

        # Save comparison
        comp_path = output_path.parent / 'losing_vs_winning_comparison.csv'
        if 'comparison' in self.results:
            self.results['comparison'].to_csv(comp_path, index=False)
            print(f"💾 Comparison saved to: {comp_path}")


def main():
    """
    Main function - Run losing streak analysis

    Usage:
        python tools/losing_streak_analyzer.py

    Or with custom CSV:
        python tools/losing_streak_analyzer.py --csv data/your_backtest.csv
    """
    import argparse

    parser = argparse.ArgumentParser(description='Losing Streak Analyzer')
    parser.add_argument('--csv', type=str,
                       default='data/backtest_GBPUSD_M15_20251026_173645.csv',
                       help='Path to backtest CSV file')
    parser.add_argument('--min-streak', type=int, default=6,
                       help='Minimum losing streak length to analyze')
    parser.add_argument('--output', type=str,
                       default='output/losing_streak_report.csv',
                       help='Output path for report')

    args = parser.parse_args()

    # Run analysis
    analyzer = LosingStreakAnalyzer(args.csv)
    results = analyzer.analyze(min_streak_length=args.min_streak)

    if results is not None:
        analyzer.save_report(args.output)

    return analyzer


if __name__ == '__main__':
    main()
