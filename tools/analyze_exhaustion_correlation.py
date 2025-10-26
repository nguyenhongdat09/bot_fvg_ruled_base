"""
Exhaustion Predictive Power Analysis

Analyzes whether exhaustion indicators have predictive power for trade outcomes.

Questions answered:
1. Do winning trades have higher exhaustion scores?
2. Is the difference statistically significant?
3. What is the optimal exhaustion threshold?
4. Should we implement exhaustion filter?

Author: Claude Code
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def analyze_exhaustion_predictive_power(csv_path):
    """
    Analyze exhaustion predictive power

    Args:
        csv_path: Path to backtest CSV file
    """

    df = pd.read_csv(csv_path)

    # Add is_win column if not exists
    if 'is_win' not in df.columns:
        df['is_win'] = (df['pnl'] > 0).astype(int)

    print("\n" + "="*80)
    print("EXHAUSTION PREDICTIVE POWER ANALYSIS")
    print("="*80)
    print(f"Dataset: {Path(csv_path).name}")
    print(f"Total Trades: {len(df)}")
    print(f"Wins: {df['is_win'].sum()} ({df['is_win'].mean()*100:.1f}%)")
    print(f"Losses: {(1-df['is_win']).sum()} ({(1-df['is_win'].mean())*100:.1f}%)")

    # ========== 1. OVERALL STATISTICS ==========
    print("\n" + "="*80)
    print("[1] OVERALL EXHAUSTION STATISTICS")
    print("="*80)

    print(f"\n📊 Exhaustion Score Distribution:")
    print(f"   Mean: {df['exhaustion_score'].mean():.2f}")
    print(f"   Median: {df['exhaustion_score'].median():.2f}")
    print(f"   Std Dev: {df['exhaustion_score'].std():.2f}")
    print(f"   Min: {df['exhaustion_score'].min():.2f}")
    print(f"   Max: {df['exhaustion_score'].max():.2f}")

    print(f"\n📊 Exhaustion Direction Distribution:")
    dir_counts = df['exhaustion_direction'].value_counts()
    for direction, count in dir_counts.items():
        pct = count / len(df) * 100
        print(f"   {direction}: {count} ({pct:.1f}%)")

    # ========== 2. WIN vs LOSS COMPARISON ==========
    print("\n" + "="*80)
    print("[2] EXHAUSTION SCORE: WIN vs LOSS COMPARISON")
    print("="*80)

    win_df = df[df['is_win'] == 1]
    loss_df = df[df['is_win'] == 0]

    win_exhaustion = win_df['exhaustion_score'].mean()
    loss_exhaustion = loss_df['exhaustion_score'].mean()
    difference = win_exhaustion - loss_exhaustion

    print(f"\n📈 Mean Exhaustion Score:")
    print(f"   Winning trades: {win_exhaustion:.2f}")
    print(f"   Losing trades: {loss_exhaustion:.2f}")
    print(f"   Difference: {difference:.2f}")

    if difference > 5:
        print(f"   ✅ Winners have HIGHER exhaustion score (GOOD SIGN!)")
    elif difference < -5:
        print(f"   ❌ Winners have LOWER exhaustion score (BAD!)")
    else:
        print(f"   ⚠️  No significant difference")

    # ========== 3. STATISTICAL SIGNIFICANCE ==========
    print("\n" + "="*80)
    print("[3] STATISTICAL SIGNIFICANCE TEST (t-test)")
    print("="*80)

    win_scores = win_df['exhaustion_score'].dropna()
    loss_scores = loss_df['exhaustion_score'].dropna()

    if len(win_scores) > 0 and len(loss_scores) > 0:
        t_stat, p_value = stats.ttest_ind(win_scores, loss_scores)

        print(f"\n📊 T-Test Results:")
        print(f"   T-statistic: {t_stat:.4f}")
        print(f"   P-value: {p_value:.4f}")

        if p_value < 0.01:
            print(f"   ✅ HIGHLY SIGNIFICANT! (p < 0.01)")
            print(f"   → Exhaustion has STRONG predictive power!")
        elif p_value < 0.05:
            print(f"   ✅ SIGNIFICANT! (p < 0.05)")
            print(f"   → Exhaustion has predictive power")
        elif p_value < 0.10:
            print(f"   🟡 MARGINALLY SIGNIFICANT (p < 0.10)")
            print(f"   → Exhaustion may have weak predictive power")
        else:
            print(f"   ❌ NOT SIGNIFICANT (p >= 0.10)")
            print(f"   → Exhaustion does NOT have predictive power")
    else:
        print("   ⚠️  Not enough data for t-test")
        p_value = 1.0

    # ========== 4. WIN RATE BY EXHAUSTION BINS ==========
    print("\n" + "="*80)
    print("[4] WIN RATE BY EXHAUSTION SCORE BINS")
    print("="*80)

    bins = [0, 30, 50, 70, 100]
    labels = ['0-30 (Low)', '30-50 (Medium)', '50-70 (High)', '70-100 (Very High)']
    df['exhaustion_bin'] = pd.cut(df['exhaustion_score'], bins=bins, labels=labels, include_lowest=True)

    print(f"\n📊 Win Rate by Exhaustion Level:")
    baseline_winrate = df['is_win'].mean() * 100

    for label in labels:
        bin_df = df[df['exhaustion_bin'] == label]
        if len(bin_df) > 0:
            win_rate = bin_df['is_win'].mean() * 100
            n_trades = len(bin_df)
            diff_from_baseline = win_rate - baseline_winrate

            indicator = "✅" if diff_from_baseline > 5 else ("❌" if diff_from_baseline < -5 else "→")

            print(f"   {indicator} {label:20s}: {win_rate:5.1f}% ({n_trades:3d} trades) [Δ{diff_from_baseline:+.1f}%]")

    print(f"\n   Baseline (all trades): {baseline_winrate:.1f}%")

    # ========== 5. COMPONENT ANALYSIS ==========
    print("\n" + "="*80)
    print("[5] COMPONENT ANALYSIS (CUSUM vs Velocity)")
    print("="*80)

    print(f"\n📊 CUSUM Score: Win vs Loss")
    win_cusum = win_df['cusum_score'].mean()
    loss_cusum = loss_df['cusum_score'].mean()
    print(f"   Winning trades: {win_cusum:.2f}")
    print(f"   Losing trades: {loss_cusum:.2f}")
    print(f"   Difference: {win_cusum - loss_cusum:.2f}")

    print(f"\n📊 Velocity Score: Win vs Loss")
    win_velocity = win_df['velocity_score'].mean()
    loss_velocity = loss_df['velocity_score'].mean()
    print(f"   Winning trades: {win_velocity:.2f}")
    print(f"   Losing trades: {loss_velocity:.2f}")
    print(f"   Difference: {win_velocity - loss_velocity:.2f}")

    # ========== 6. DIRECTION ANALYSIS ==========
    print("\n" + "="*80)
    print("[6] EXHAUSTION DIRECTION ANALYSIS")
    print("="*80)

    print(f"\n📊 Win Rate by Exhaustion Direction:")

    for direction in ['bullish_exhaustion', 'bearish_exhaustion', 'none']:
        dir_df = df[df['exhaustion_direction'] == direction]
        if len(dir_df) > 0:
            win_rate = dir_df['is_win'].mean() * 100
            n_trades = len(dir_df)
            diff = win_rate - baseline_winrate

            indicator = "✅" if diff > 5 else ("❌" if diff < -5 else "→")
            print(f"   {indicator} {direction:20s}: {win_rate:5.1f}% ({n_trades:3d} trades) [Δ{diff:+.1f}%]")

    # ========== 7. CORRELATION WITH PNL ==========
    print("\n" + "="*80)
    print("[7] CORRELATION WITH PNL")
    print("="*80)

    corr_exhaustion = df[['exhaustion_score', 'pnl']].corr().iloc[0, 1]
    corr_cusum = df[['cusum_score', 'pnl']].corr().iloc[0, 1]
    corr_velocity = df[['velocity_score', 'pnl']].corr().iloc[0, 1]

    print(f"\n📊 Correlation with PnL:")
    print(f"   Exhaustion Score: {corr_exhaustion:.4f}")
    print(f"   CUSUM Score: {corr_cusum:.4f}")
    print(f"   Velocity Score: {corr_velocity:.4f}")

    if abs(corr_exhaustion) > 0.1:
        print(f"   ✅ Moderate correlation detected!")
    elif abs(corr_exhaustion) > 0.05:
        print(f"   🟡 Weak correlation")
    else:
        print(f"   ❌ No meaningful correlation")

    # ========== 8. OPTIMAL THRESHOLD ==========
    print("\n" + "="*80)
    print("[8] OPTIMAL EXHAUSTION THRESHOLD ANALYSIS")
    print("="*80)

    print(f"\n📊 Testing different thresholds:")
    thresholds = [40, 50, 60, 70, 80]

    best_threshold = 70
    best_improvement = -999

    for threshold in thresholds:
        high_exhaust = df[df['exhaustion_score'] >= threshold]
        if len(high_exhaust) > 10:  # Enough samples
            wr = high_exhaust['is_win'].mean() * 100
            n = len(high_exhaust)
            improvement = wr - baseline_winrate

            indicator = "✅" if improvement > 5 else ("🟡" if improvement > 0 else "❌")
            print(f"   {indicator} Threshold ≥{threshold}: {wr:.1f}% WR ({n} trades) [Δ{improvement:+.1f}%]")

            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = threshold

    # ========== 9. FINAL RECOMMENDATION ==========
    print("\n" + "="*80)
    print("[9] 🎯 FINAL RECOMMENDATION")
    print("="*80)

    # Decision logic
    should_implement = False
    recommendation = ""

    if p_value < 0.05 and difference > 5:
        should_implement = True
        recommendation = "IMPLEMENT EXHAUSTION FILTER"
        reason = f"Exhaustion has SIGNIFICANT predictive power (p={p_value:.4f})"
    elif p_value < 0.10 and difference > 3:
        should_implement = True
        recommendation = "IMPLEMENT WITH CAUTION"
        reason = f"Exhaustion shows WEAK predictive power (p={p_value:.4f})"
    else:
        should_implement = False
        recommendation = "DO NOT IMPLEMENT"
        reason = f"Exhaustion has NO significant predictive power (p={p_value:.4f})"

    print(f"\n{'✅' if should_implement else '❌'} RECOMMENDATION: {recommendation}")
    print(f"   Reason: {reason}")

    if should_implement:
        print(f"\n📋 Implementation Details:")
        print(f"   1. Add filter: exhaustion_score >= {best_threshold}")
        print(f"   2. Expected improvement: +{best_improvement:.1f}% win rate")
        print(f"   3. Trade reduction: ~{(1 - len(df[df['exhaustion_score'] >= best_threshold])/len(df))*100:.1f}%")
        print(f"\n   Next steps:")
        print(f"   → Run: Implement exhaustion filter in strategy")
        print(f"   → Re-test and validate improvement")
    else:
        print(f"\n📋 Alternative Approaches:")
        print(f"   1. Try combining exhaustion with other filters")
        print(f"   2. Tune CUSUM/velocity parameters")
        print(f"   3. Focus on adaptive risk management instead")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

    return {
        'should_implement': should_implement,
        'p_value': p_value,
        'win_loss_difference': difference,
        'best_threshold': best_threshold,
        'best_improvement': best_improvement
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Find latest CSV
        data_dir = Path('data')
        csvs = sorted(data_dir.glob('backtest_*.csv'), key=lambda x: x.stat().st_mtime, reverse=True)

        if not csvs:
            print("ERROR: No backtest CSV files found in data/")
            sys.exit(1)

        csv_path = csvs[0]
        print(f"Using latest CSV: {csv_path.name}")

    result = analyze_exhaustion_predictive_power(csv_path)
