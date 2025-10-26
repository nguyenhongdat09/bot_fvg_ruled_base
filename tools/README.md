# Trading Strategy Analysis Tools

Comprehensive toolkit for analyzing and optimizing FVG trading strategies through data-driven insights.

---

## 📦 Available Tools

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| **CSV Merger** | Combine multiple backtest periods | Multiple CSV files | Single merged CSV |
| **Feature Optimization** | Identify important indicators | Backtest CSV | Feature ranking + recommendations |
| **Losing Streak Analyzer** | Find patterns in losing streaks | Backtest CSV | Statistical insights + filters |

---

## 🔄 Recommended Workflow

```
1. Merge CSVs (if you have multiple backtest periods)
   ↓
2. Run Feature Optimization (identify important indicators)
   ↓
3. Update config.py with optimized weights
   ↓
4. Run backtest with new config
   ↓
5. Run Losing Streak Analyzer (find patterns, add filters)
   ↓
6. Test again and iterate
```

---

# 1️⃣ CSV Merger Tool

## 🎯 Purpose

Combine multiple backtest CSV files from different time periods into a single dataset for robust statistical analysis.

## ✨ Features

- **Auto-detect** all `backtest_*.csv` files in folder
- **Validate** column consistency across files
- **Remove duplicates** based on entry_time
- **Sort chronologically** by entry_time
- **Add metadata** (source_file column)
- **Display statistics** (win rate, PnL, mode distribution, etc.)

## 🚀 Usage

### Basic Usage

```bash
python tools/merge_backtest_csvs.py
```

This will:
1. Find all `backtest_*.csv` files in `data/` folder
2. Merge them into one dataset
3. Save to `data/merged_backtest.csv`

### Custom Parameters

```bash
python tools/merge_backtest_csvs.py \
  --folder data \
  --pattern 'backtest_GBPUSD_*.csv' \
  --output data/merged_gbpusd.csv
```

### Parameters

- `--folder`: Folder containing CSV files (default: `data`)
- `--pattern`: File pattern to match (default: `backtest_*.csv`)
- `--output`: Output file path (default: `data/merged_backtest.csv`)

## 📊 Example Output

```
================================================================================
BACKTEST CSV MERGER
================================================================================

[STEP 1] Finding CSV files in: data
   Pattern: backtest_*.csv
--------------------------------------------------------------------------------
✓ Found 3 CSV files:
   1. backtest_GBPUSD_M15_20251020_120000.csv        ( 412 trades)
   2. backtest_GBPUSD_M15_20251023_143000.csv        ( 298 trades)
   3. backtest_GBPUSD_M15_20251026_162732.csv        ( 877 trades)

[STEP 2] Validating Column Consistency
--------------------------------------------------------------------------------
✓ Reference columns (59): backtest_GBPUSD_M15_20251020_120000.csv
✓ All 3 CSV files have compatible columns!

[STEP 3] Merging CSV Files
--------------------------------------------------------------------------------
✓ Loaded backtest_GBPUSD_M15_20251020_120000.csv        412 trades
✓ Loaded backtest_GBPUSD_M15_20251023_143000.csv        298 trades
✓ Loaded backtest_GBPUSD_M15_20251026_162732.csv        877 trades

✓ Concatenating 3 dataframes...
✓ Total trades before dedup: 1587

✓ Removing duplicates (based on entry_time)...
⚠️  Removed 142 duplicate trades
✓ Total trades after dedup: 1445

✓ Sorting by entry_time...
✓ Date range: 2023-04-12 08:45:00 to 2025-10-24 15:30:00

[SUMMARY] Merged Dataset Statistics
================================================================================

📊 BASIC STATISTICS:
   Total Trades: 1445
   Total Columns: 60

💰 PERFORMANCE:
   Wins: 493 (34.1%)
   Losses: 952 (65.9%)
   Total PnL: $8,234.50
   Avg Win: $45.23
   Avg Loss: -$18.67
   Profit Factor: 2.42

📈 MODE DISTRIBUTION:
   REAL: 1445 (100.0%)

↕️  DIRECTION DISTRIBUTION:
   LONG: 723 (50.0%)
   SHORT: 722 (50.0%)

📁 SOURCE FILES:
   backtest_GBPUSD_M15_20251026_162732.csv: 877 (60.7%)
   backtest_GBPUSD_M15_20251020_120000.csv: 412 (28.5%)
   backtest_GBPUSD_M15_20251023_143000.csv: 298 (20.6%)

📅 DATE RANGE:
   Start: 2023-04-12 08:45:00
   End: 2025-10-24 15:30:00
   Duration: 926 days

💾 Merged data saved to: data/merged_backtest.csv
   Total trades: 1445
   File size: 0.42 MB

================================================================================
✅ MERGE COMPLETED SUCCESSFULLY!
================================================================================

🎉 SUCCESS! You can now use merged data for analysis:

   # Feature Optimization Pipeline:
   python tools/feature_optimization_pipeline.py --csv data/merged_backtest.csv

   # Losing Streak Analyzer:
   python tools/losing_streak_analyzer.py --csv data/merged_backtest.csv
```

## 💡 When to Use

- You have backtests from multiple time periods (different --days runs)
- You want more statistical significance (more samples = better analysis)
- You want to validate patterns across different market conditions
- You want to feed robust data to Feature Optimization or Losing Streak Analyzer

## 🔧 Advanced Usage

```python
from tools.merge_backtest_csvs import BacktestCSVMerger

# Initialize
merger = BacktestCSVMerger('data')

# Find CSVs
merger.find_csv_files('backtest_GBPUSD_*.csv')

# Validate columns
merger.validate_columns()

# Merge
merged_df = merger.merge_all(remove_duplicates=True)

# Print summary
merger.print_summary()

# Save
merger.save('data/my_merged.csv')
```

---

# 2️⃣ Feature Optimization Pipeline

## 🎯 Purpose

Analyze and rank features to identify the most important indicators for trading strategy performance using machine learning.

## 📋 Pipeline Steps

### 1. **Correlation Analysis**
- Identifies features with high correlation (>0.85)
- Removes redundant features
- Output: List of highly correlated pairs

### 2. **VIF Analysis (Variance Inflation Factor)**
- Detects multicollinearity (VIF >10)
- Finds features that can be predicted by others
- Output: VIF scores for each feature

### 3. **Walk-Forward Validation**
- Tests feature stability over time using TimeSeriesSplit
- Simulates real trading conditions (no look-ahead bias)
- Output: Performance metrics across 5 time windows

### 4. **Permutation Importance**
- Shuffles each feature and measures performance drop
- Uses XGBoost as backbone model
- Output: Feature importance ranking

### 5. **Ablation Study**
- Removes each feature one at a time
- Measures impact on model performance
- Output: Performance drop when feature is removed

### 6. **SHAP Values**
- Explains individual predictions
- Shows feature contribution to each trade
- Output: SHAP importance scores (with graceful fallback if XGBoost incompatible)

### 7. **Final Ranking**
- Combines all analyses into consensus ranking
- Provides KEEP/REMOVE/OPTIONAL recommendations
- Output: `feature_ranking.csv`

## 🚀 Usage

### Basic Usage

```bash
python tools/feature_optimization_pipeline.py
```

This will:
1. Load the latest backtest CSV from `data/` folder
2. Run full 7-step pipeline
3. Save results to: `output/feature_ranking.csv`

### Custom CSV

```bash
python tools/feature_optimization_pipeline.py --csv data/merged_backtest.csv
```

### Custom Output Path

```bash
python tools/feature_optimization_pipeline.py \
  --csv data/backtest.csv \
  --output results/my_ranking.csv
```

## 📊 Input Requirements

### CSV Structure

Your backtest CSV must contain these columns:

**Raw Indicator Values:**
- `hurst` - Hurst Exponent (0-1)
- `lr_deviation` - Linear Regression deviation (σ)
- `r2` - R-squared (0-1)
- `skewness` - Distribution skewness
- `kurtosis` - Distribution kurtosis
- `obv_divergence` - OBV divergence signal
- `atr_percentile` - ATR percentile (0-100)

**Component Scores:**
- `score_fvg` - FVG score
- `score_fvg_size_atr` - FVG size/ATR score
- `score_hurst` - Hurst score
- `score_lr_deviation` - LR deviation score
- `score_skewness` - Skewness score
- `score_kurtosis` - Kurtosis score
- `score_obv_div` - OBV divergence score
- `score_regime` - Regime penalty

**Target:**
- `pnl` - Profit/Loss (will be converted to win/loss binary)

### Minimum Data

- **Samples**: ≥500 (recommended: ≥800)
- **Features**: 10-20
- **Sample/Feature ratio**: ≥10:1

## 📈 Output

### feature_ranking.csv

| Column | Description |
|--------|-------------|
| `Feature` | Feature name |
| `Importance` | Permutation importance score |
| `Perm_rank` | Rank by permutation importance |
| `Performance_drop` | Drop when feature removed (ablation) |
| `Ablation_rank` | Rank by ablation study |
| `SHAP_importance` | Mean absolute SHAP value |
| `SHAP_rank` | Rank by SHAP values |
| `Avg_rank` | Average rank across methods |
| `High_correlation` | Flag for high correlation |
| `High_VIF` | Flag for multicollinearity |
| `Recommendation` | KEEP / REMOVE / OPTIONAL |

### Recommendations

- **KEEP**: Top 8 features, no correlation/VIF issues
- **REMOVE**: High correlation or VIF (multicollinear)
- **OPTIONAL**: Lower importance, can test enabling/disabling

### Example Output

```
[STEP 8] Generating Final Feature Ranking
================================================================================

✅ FINAL FEATURE RANKING:
           Feature  Importance  Perm_rank  Performance_drop  Ablation_rank  SHAP_importance  SHAP_rank  Avg_rank  High_correlation  High_VIF Recommendation
0     lr_deviation       0.0136          1             0.028              1            0.0136          1      1.00             False     False           KEEP
1         kurtosis       0.0034          2            -0.006             14            0.0034          2      6.00             False     False           KEEP
2     score_regime       0.0000          3             0.000             13            0.0000          3      6.33             False     False           KEEP
3  score_fvg_size_atr    0.0000          6             0.000             11            0.0000          6      7.67             False     False           KEEP

📊 SUMMARY:
   KEEP: 8 features
   REMOVE: 1 feature (score_kurtosis - HIGH CORRELATION)
   OPTIONAL: 6 features

✅ RECOMMENDED FEATURES TO KEEP:
   1. lr_deviation (Importance: 0.0136, Ablation: +0.028)
   2. kurtosis
   3. score_regime
   4. score_hurst
   5. score_fvg
   6. score_fvg_size_atr
   7. score_skewness
   8. obv_divergence

⚠️  RECOMMENDED TO REMOVE:
   1. score_kurtosis (HIGH CORRELATION with kurtosis)

📝 NEXT STEPS:
   1. Update config.py confluence_weights based on recommendations
   2. Increase weights for top KEEP features
   3. Remove or reduce REMOVE features
   4. Re-run backtest and compare performance
```

## 🔬 Algorithm Details

### XGBoost Configuration

```python
XGBClassifier(
    max_depth=3,          # Prevent overfitting
    n_estimators=50-100,  # Fast training
    learning_rate=0.1,    # Standard rate
    random_state=42       # Reproducibility
)
```

**Why these params?**
- `max_depth=3`: Simple trees to avoid overfitting with ~800 samples
- Focus on **feature selection**, not model optimization

### Time-Series Split

```python
TimeSeriesSplit(n_splits=5)
```

**Splits for 877 samples:**
- Fold 1: Train [0:175], Test [175:350]
- Fold 2: Train [0:350], Test [350:525]
- Fold 3: Train [0:525], Test [525:700]
- Fold 4: Train [0:700], Test [700:875]
- Fold 5: Train [0:875], Test [875:877]

**Why Time-Series Split?**
- No shuffle → respects temporal order
- No look-ahead bias → realistic evaluation
- Simulates walk-forward testing

## ⚙️ Dependencies

```bash
pip install xgboost shap statsmodels scikit-learn pandas numpy matplotlib seaborn scipy
```

## 🔧 Troubleshooting

### Error: "Missing features in CSV"

**Solution:** Re-run backtest with updated code that exports raw indicators and component scores.

```bash
# Pull latest code
git pull origin claude/read-document-project-011CUQHRGnqfNqLViNGpmb8S

# Re-run backtest
python examples/run_backtest.py --days 730
```

### Error: "Not enough data"

**Solution:** Pipeline needs ≥500 samples. If you have fewer trades:
1. Lower confidence threshold to generate more signals
2. Merge multiple CSVs using the CSV Merger tool
3. Run longer backtest period (--days 1000)

### Warning: "High VIF features"

This is **normal**! Some features naturally correlate (e.g., `hurst` and `score_hurst`). The pipeline will recommend which to keep.

### Warning: "SHAP analysis failed"

This is **expected** with XGBoost 3.x compatibility issues. Pipeline gracefully falls back to Permutation Importance. You still get complete feature rankings.

## 🎓 Advanced Usage

### Run in Python Script

```python
from tools.feature_optimization_pipeline import FeatureOptimizationPipeline

# Initialize
pipeline = FeatureOptimizationPipeline('data/backtest.csv')

# Run pipeline
results = pipeline.run()

# Access individual results
corr_results = pipeline.results['correlation']
vif_results = pipeline.results['vif']
perm_importance = pipeline.results['permutation_importance']

# Save report
pipeline.save_report('output/ranking.csv')
```

### Customize Thresholds

```python
# Modify correlation threshold
pipeline.correlation_analysis(threshold=0.90)  # More lenient

# Modify VIF threshold
pipeline.vif_analysis(threshold=5.0)  # More strict
```

---

# 3️⃣ Losing Streak Analyzer

## 🎯 Purpose

**CRITICAL TOOL** for risk management! Identifies common patterns in long losing streaks to help you reduce maximum drawdown.

> "Vấn đề không hoàn toàn là winrate mà còn nằm ở losing streak có những streak lên tận 10 lần khó mà chấp nhận được trong khi tôi chỉ muốn 6 lần là max rồi"

This tool helps you answer:
- What market conditions cause long losing streaks?
- How are losing streaks different from winning streaks?
- Which filters can reduce losing streak length?

## ✨ Features

- **Identify all streaks** (winning and losing)
- **Analyze long losing streaks** in detail (≥6 trades by default)
- **Statistical comparison** (t-tests) between losing vs winning streaks
- **Feature distribution analysis** (volatility, Hurst, skewness, etc.)
- **Actionable recommendations** with expected impact
- **Detailed trade breakdowns** for each long losing streak

## 🚀 Usage

### Basic Usage

```bash
python tools/losing_streak_analyzer.py
```

This will:
1. Load the latest backtest CSV from `data/` folder
2. Analyze losing streaks ≥6 trades
3. Save results to: `output/losing_streak_analysis.csv`

### Custom Parameters

```bash
python tools/losing_streak_analyzer.py \
  --csv data/merged_backtest.csv \
  --min-streak 8 \
  --output output/my_analysis.csv
```

### Parameters

- `--csv`: CSV file to analyze (default: latest in `data/`)
- `--min-streak`: Minimum streak length to analyze (default: `6`)
- `--output`: Output file path (default: `output/losing_streak_analysis.csv`)

## 📊 Example Output

```
================================================================================
LOSING STREAK ANALYZER
================================================================================

Dataset: data/merged_backtest.csv
Total Trades: 1445
Win Rate: 34.1%

[STEP 1] Identifying Streaks (minimum length: 1)
--------------------------------------------------------------------------------
✓ Found 287 streaks total:
   - Losing streaks: 165 (57.5%)
   - Winning streaks: 122 (42.5%)

📊 LOSING STREAK DISTRIBUTION:
   Length 1: 45 streaks (27.3%)
   Length 2: 38 streaks (23.0%)
   Length 3: 29 streaks (17.6%)
   Length 4: 21 streaks (12.7%)
   Length 5: 15 streaks (9.1%)
   Length 6: 9 streaks (5.5%)
   Length 7: 4 streaks (2.4%)
   Length 8: 2 streaks (1.2%)
   Length 9: 1 streak (0.6%)
   Length 10: 1 streak (0.6%)

⚠️  MAXIMUM LOSING STREAK: 10 trades
⚠️  MAXIMUM WINNING STREAK: 7 trades

[STEP 2] Analyzing Long Losing Streaks (≥6 trades)
--------------------------------------------------------------------------------
✓ Found 17 long losing streaks (≥6 trades)
✓ Total trades in long losing streaks: 114 (7.9% of all trades)

🔴 LONG LOSING STREAKS BREAKDOWN:

Streak #1: 10 trades (2024-08-15 to 2024-08-22)
   Entry times: 2024-08-15 08:30, 2024-08-16 12:45, ... [10 total]
   Avg Hurst: 0.38 (mean reverting!)
   Avg ATR Percentile: 78.2 (HIGH volatility!)
   Avg Skewness: -0.42
   Total Loss: -$482.50

Streak #2: 8 trades (2024-09-03 to 2024-09-07)
   Entry times: 2024-09-03 14:20, 2024-09-04 09:15, ... [8 total]
   Avg Hurst: 0.41 (mean reverting!)
   Avg ATR Percentile: 72.5 (HIGH volatility!)
   Avg Skewness: 0.15
   Total Loss: -$368.20

[... more streaks ...]

[STEP 3] Comparing with Winning Streaks
--------------------------------------------------------------------------------
✓ Comparing losing streaks (≥6) vs winning streaks (≥3)
✓ Losing streak trades: 114
✓ Winning streak trades: 187

🔍 STATISTICAL COMPARISON (t-tests):

Feature: hurst
   Losing streaks avg: 0.42 ± 0.08
   Winning streaks avg: 0.58 ± 0.12
   Difference: -0.16 (SIGNIFICANT, p=0.0001)
   ⚠️  Losing streaks occur in MEAN REVERTING conditions (Hurst < 0.5)!

Feature: atr_percentile
   Losing streaks avg: 74.3 ± 15.2
   Winning streaks avg: 45.8 ± 18.7
   Difference: +28.5 (SIGNIFICANT, p<0.0001)
   ⚠️  Losing streaks occur in HIGH VOLATILITY conditions!

Feature: lr_deviation
   Losing streaks avg: 2.8 ± 0.6
   Winning streaks avg: 1.9 ± 0.5
   Difference: +0.9 (SIGNIFICANT, p=0.003)
   ⚠️  Losing streaks occur when price is FAR from fair value!

Feature: skewness
   Losing streaks avg: -0.12 ± 0.45
   Winning streaks avg: 0.08 ± 0.38
   Difference: -0.20 (p=0.082, not significant)

Feature: r2
   Losing streaks avg: 0.45 ± 0.18
   Winning streaks avg: 0.68 ± 0.15
   Difference: -0.23 (SIGNIFICANT, p=0.001)
   ⚠️  Losing streaks occur in CHOPPY markets (low R²)!

[STEP 4] Generating Recommendations
--------------------------------------------------------------------------------

✅ ACTIONABLE RECOMMENDATIONS TO REDUCE LOSING STREAKS:

1. ADD FILTER: Skip trades when Hurst < 0.45
   - Rationale: 73% of losing streak trades had Hurst < 0.45
   - Expected Impact: Could avoid 83/114 (72.8%) of losing streak trades
   - Implementation:
     ```python
     # In fvg_confluence_strategy.py, analyze() method:
     if hurst < 0.45:
         return {'signal': 0, 'reason': 'Hurst too low (mean reverting)'}
     ```

2. ADD FILTER: Skip trades when ATR Percentile > 70
   - Rationale: 68% of losing streak trades had ATR > 70
   - Expected Impact: Could avoid 78/114 (68.4%) of losing streak trades
   - Implementation:
     ```python
     if atr_percentile > 70:
         return {'signal': 0, 'reason': 'ATR too high (excessive volatility)'}
     ```

3. ADD FILTER: Skip trades when R² < 0.50
   - Rationale: Low R² indicates choppy, non-trending conditions
   - Expected Impact: Could avoid 67/114 (58.8%) of losing streak trades
   - Implementation:
     ```python
     if r2 < 0.50:
         return {'signal': 0, 'reason': 'R² too low (choppy market)'}
     ```

4. INCREASE WEIGHT: Increase lr_deviation weight in confluence scoring
   - Rationale: lr_deviation shows strong predictive power
   - Current weight: 25%
   - Suggested: 30-35%

📊 COMBINED FILTER ANALYSIS:

If you apply filters #1 AND #2 (Hurst < 0.45 OR ATR > 70):
   - Trades avoided: 94/114 (82.5%) of losing streak trades
   - But also removes: 23/187 (12.3%) of winning streak trades
   - Net benefit: Significant reduction in losing streaks!

⚠️  WARNING: Aggressive filtering reduces trade frequency!
   - Current: 1445 trades over 926 days (1.56 trades/day)
   - After filtering: ~1200 trades (1.30 trades/day)
   - Make sure you're comfortable with reduced trade frequency

[STEP 5] Saving Results
--------------------------------------------------------------------------------
💾 Detailed analysis saved to: output/losing_streak_analysis.csv

================================================================================
✅ ANALYSIS COMPLETE!
================================================================================

📝 NEXT STEPS:
1. Review recommendations above
2. Implement 1-2 filters at a time (start conservative)
3. Re-run backtest and check:
   - Maximum losing streak reduced?
   - Win rate improved?
   - Trade frequency still acceptable?
4. Iterate and fine-tune thresholds
```

## 📈 Output Files

### losing_streak_analysis.csv

Contains detailed breakdown of each long losing streak:

| Column | Description |
|--------|-------------|
| `streak_id` | Unique streak identifier |
| `length` | Number of consecutive losses |
| `start_date` | First trade in streak |
| `end_date` | Last trade in streak |
| `total_loss` | Cumulative loss ($) |
| `avg_hurst` | Average Hurst in streak |
| `avg_atr_percentile` | Average ATR percentile |
| `avg_lr_deviation` | Average LR deviation |
| `avg_skewness` | Average skewness |
| `avg_r2` | Average R² |
| `trade_indices` | List of trade row numbers |

## 💡 When to Use

- After optimizing features (step 2 in workflow)
- When maximum losing streak is unacceptable (>6-8 trades)
- To understand WHY losing streaks happen
- To find specific market conditions to avoid
- Before adding new filters (data-driven decision)

## 🔧 Advanced Usage

### Run in Python Script

```python
from tools.losing_streak_analyzer import LosingStreakAnalyzer

# Initialize
analyzer = LosingStreakAnalyzer('data/merged_backtest.csv')

# Find all streaks
analyzer.identify_streaks(min_length=1)

# Analyze long losing streaks
analyzer.analyze_long_losing_streaks(min_streak_length=6)

# Compare with winners
analyzer.compare_with_winners()

# Generate recommendations
analyzer.generate_recommendations()

# Save results
analyzer.save_results('output/my_analysis.csv')

# Print summary
analyzer.print_summary()
```

### Custom Analysis

```python
# Analyze very long streaks only
analyzer.analyze_long_losing_streaks(min_streak_length=8)

# Compare specific features
features = ['hurst', 'atr_percentile', 'custom_indicator']
analyzer.compare_features(features)

# Access raw data
all_streaks = analyzer.streaks
long_losing_streaks = analyzer.long_losing_streaks
```

---

## 📚 References

- **Permutation Importance**: Breiman, L. (2001). Random Forests
- **SHAP Values**: Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions
- **VIF**: Belsley, Kuh, & Welsch (1980). Regression Diagnostics
- **Walk-Forward**: Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies
- **Hurst Exponent**: Mandelbrot & Wallis (1968). Noah, Joseph, and Operational Hydrology

---

## 🔄 Complete Workflow Example

Here's a complete workflow for optimizing your FVG trading strategy:

### Step 1: Merge Multiple Backtests

```bash
# If you have multiple backtest CSVs
python tools/merge_backtest_csvs.py

# Output: data/merged_backtest.csv (1445 trades)
```

### Step 2: Optimize Features

```bash
# Identify important features
python tools/feature_optimization_pipeline.py --csv data/merged_backtest.csv

# Review: output/feature_ranking.csv
```

### Step 3: Update Config

Based on feature_ranking.csv, update `config.py`:

```python
'confluence_weights': {
    'fvg': 50,              # Core strategy (keep high)
    'fvg_size_atr': 15,     # Quality filter
    'lr_deviation': 30,     # TOP FEATURE (increase from 25%)
    'skewness': 10,         # Useful filter
    'hurst': 0,             # REMOVE (negative importance)
    'kurtosis': 0,          # REMOVE
    'obv_div': 0,           # REMOVE
    'regime': 0,            # REMOVE
}
```

### Step 4: Run Backtest with New Config

```bash
python examples/run_backtest.py --days 730

# Output: data/backtest_GBPUSD_M15_[timestamp].csv
```

### Step 5: Analyze Losing Streaks

```bash
# Find patterns in long losing streaks
python tools/losing_streak_analyzer.py \
  --csv data/backtest_GBPUSD_M15_[timestamp].csv \
  --min-streak 6

# Review: output/losing_streak_analysis.csv
```

### Step 6: Add Filters

Based on losing_streak_analysis.csv recommendations, add filters in `strategies/fvg_confluence_strategy.py`:

```python
def analyze(self, index: int) -> dict:
    # ... existing code ...

    # Losing streak filters (based on analysis)
    hurst = self.data.iloc[index]['hurst']
    if hurst < 0.45:
        return {'signal': 0, 'reason': 'Hurst too low (mean reverting)'}

    atr_percentile = self.data.iloc[index]['ATR_percentile']
    if atr_percentile > 70:
        return {'signal': 0, 'reason': 'ATR too high (excessive volatility)'}

    r2 = self.data.iloc[index]['r2']
    if r2 < 0.50:
        return {'signal': 0, 'reason': 'R² too low (choppy market)'}

    # ... continue with confluence scoring ...
```

### Step 7: Test and Iterate

```bash
# Re-run backtest with new filters
python examples/run_backtest.py --days 730

# Compare results:
# - Maximum losing streak: 10 → 6 ✅
# - Win rate: 34% → 38% ✅
# - Trade frequency: 1.56/day → 1.30/day (acceptable)
# - Profit factor: 2.03 → 2.45 ✅
```

---

## 📧 Support

For issues or questions:

1. **Check CSV has all required columns** (59 columns including raw indicators + component scores)
2. **Ensure sufficient data** (≥500 trades for meaningful analysis)
3. **Review error messages** carefully - most issues are data-related
4. **Start with merged CSV** if you have multiple backtests (more robust results)

---

## 📝 Key Insights from Analysis

### Feature Optimization Results

Based on 877-trade backtest:
- **Top features**: lr_deviation (0.0136), kurtosis (0.0034)
- **Zero importance**: FVG features (expected - all trades have FVG)
- **Negative importance**: Many features add noise
- **Recommendation**: Simplify from 9 features to 4 core features

### Losing Streak Patterns

Common patterns in long losing streaks (≥6 trades):
- **Low Hurst** (H < 0.45): Mean reverting conditions
- **High ATR** (>70th percentile): Excessive volatility
- **Low R²** (<0.5): Choppy, non-trending markets
- **High LR deviation** (>2.5σ): Price too far from fair value

### Risk Management Priority

> **Win rate is NOT everything!**
>
> A 40% win rate with max 6-streak is BETTER than a 45% win rate with max 12-streak.
>
> Focus on:
> 1. Reducing maximum losing streak length
> 2. Maintaining acceptable trade frequency
> 3. Keeping profit factor >2.0

---

**Author**: Claude Code
**Date**: 2025-10-26
**Version**: 2.0

---

## 🎯 Quick Command Reference

```bash
# Merge multiple CSVs
python tools/merge_backtest_csvs.py

# Optimize features
python tools/feature_optimization_pipeline.py --csv data/merged_backtest.csv

# Analyze losing streaks
python tools/losing_streak_analyzer.py --csv data/merged_backtest.csv --min-streak 6

# Run backtest
python examples/run_backtest.py --days 730
```

**Remember**: Always use merged CSVs when possible for more robust statistical analysis!
