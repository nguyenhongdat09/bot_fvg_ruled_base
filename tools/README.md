# Feature Optimization Pipeline

Comprehensive ML pipeline for feature engineering and selection in trading strategies.

## 🎯 Purpose

Analyze and rank features to identify the most important indicators for trading strategy performance.

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
- Output: SHAP importance scores

### 7. **Final Ranking**
- Combines all analyses into consensus ranking
- Provides KEEP/REMOVE/OPTIONAL recommendations
- Output: `feature_ranking.csv`

---

## 🚀 Usage

### Basic Usage

```bash
python tools/feature_optimization_pipeline.py
```

This will:
1. Load the default CSV: `data/backtest_GBPUSD_M15_20251026_162732.csv`
2. Run full pipeline
3. Save results to: `output/feature_ranking.csv`

### Custom CSV

```bash
python tools/feature_optimization_pipeline.py --csv data/your_backtest.csv
```

### Custom Output Path

```bash
python tools/feature_optimization_pipeline.py \
  --csv data/backtest.csv \
  --output results/my_ranking.csv
```

---

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

---

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

---

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

**Splits for 866 samples:**
- Fold 1: Train [0:173], Test [173:346]
- Fold 2: Train [0:346], Test [346:519]
- Fold 3: Train [0:519], Test [519:692]
- Fold 4: Train [0:692], Test [692:865]
- Fold 5: Train [0:865], Test [865:866]

**Why Time-Series Split?**
- No shuffle → respects temporal order
- No look-ahead bias → realistic evaluation
- Simulates walk-forward testing

---

## 📊 Example Output

```
[STEP 8] Generating Final Feature Ranking
================================================================================

✅ FINAL FEATURE RANKING:
           Feature  Importance  Perm_rank  Performance_drop  Ablation_rank  SHAP_importance  SHAP_rank  Avg_rank  High_correlation  High_VIF Recommendation
0  score_lr_deviation       0.045          1             0.032              1            0.082          1      1.00             False     False           KEEP
1           score_fvg       0.038          2             0.028              2            0.075          2      2.00             False     False           KEEP
2  score_fvg_size_atr       0.031          3             0.021              3            0.061          3      3.00             False     False           KEEP
3       lr_deviation       0.025          4             0.018              4            0.052          4      4.00             False     False           KEEP
4              hurst       0.022          5             0.015              5            0.048          5      5.00             False     False           KEEP

📊 SUMMARY:
   KEEP: 8 features
   REMOVE: 2 features
   OPTIONAL: 4 features

✅ RECOMMENDED FEATURES TO KEEP:
   1. score_lr_deviation
   2. score_fvg
   3. score_fvg_size_atr
   4. lr_deviation
   5. hurst
   6. score_hurst
   7. r2
   8. skewness
```

---

## ⚙️ Dependencies

```bash
pip install xgboost shap statsmodels scikit-learn pandas numpy matplotlib seaborn
```

---

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

**Solution:** Pipeline needs ≥500 samples. If you have fewer trades, lower the confidence threshold to generate more signals.

```python
# In config.py
'min_confidence_score': 60.0  # Lower from 70.0
```

### Warning: "High VIF features"

This is **normal**! Some features naturally correlate (e.g., `hurst` and `score_hurst`). The pipeline will recommend which to keep.

---

## 📝 Next Steps After Running Pipeline

1. **Review `feature_ranking.csv`**
   - Check which features got KEEP recommendation
   - Understand why some were flagged for REMOVE

2. **Update strategy weights**
   - Increase weights for KEEP features
   - Decrease/remove REMOVE features

3. **Re-run backtest**
   - Test with optimized feature set
   - Compare performance

4. **Iterate**
   - Try different feature combinations
   - Test on longer time periods

---

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

## 📚 References

- **Permutation Importance**: Breiman, L. (2001). Random Forests
- **SHAP Values**: Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions
- **VIF**: Belsley, Kuh, & Welsch (1980). Regression Diagnostics
- **Walk-Forward**: Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies

---

## 📧 Support

For issues or questions:
1. Check CSV has all required columns
2. Ensure ≥500 samples
3. Review error messages carefully

---

**Author**: Claude Code
**Date**: 2025-10-26
**Version**: 1.0
