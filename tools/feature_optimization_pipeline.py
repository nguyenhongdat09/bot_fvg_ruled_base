"""
Feature Optimization Pipeline for Trading Strategy

Comprehensive pipeline to evaluate and rank features:
1. Correlation Analysis - Remove highly correlated features (>0.85)
2. VIF Analysis - Detect multicollinearity (VIF >10)
3. Walk-Forward Validation - Test temporal stability
4. Permutation Importance - XGBoost feature ranking
5. Ablation Study - Measure impact of removing each feature
6. SHAP Values - Explain feature contributions

Output: feature_ranking.csv with Keep/Remove recommendations

Author: Claude Code
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureOptimizationPipeline:
    """
    Feature Optimization Pipeline for Trading Data

    Usage:
        pipeline = FeatureOptimizationPipeline('data/backtest.csv')
        results = pipeline.run()
        pipeline.save_report('output/feature_ranking.csv')
    """

    def __init__(self, csv_path: str, target_col: str = 'pnl'):
        """
        Initialize pipeline

        Args:
            csv_path: Path to backtest CSV
            target_col: Target column name (default: 'pnl')
        """
        self.csv_path = Path(csv_path)
        self.target_col = target_col

        # Feature columns (statistical indicators)
        self.feature_cols = [
            # Raw indicator values
            'hurst',
            'lr_deviation',
            'r2',
            'skewness',
            'kurtosis',
            'obv_divergence',
            'atr_percentile',
            # Component scores
            'score_fvg',
            'score_fvg_size_atr',
            'score_hurst',
            'score_lr_deviation',
            'score_skewness',
            'score_kurtosis',
            'score_obv_div',
            'score_regime',
        ]

        # Results storage
        self.results = {}
        self.df = None
        self.X = None
        self.y = None

        print("=" * 80)
        print("FEATURE OPTIMIZATION PIPELINE")
        print("=" * 80)

    def load_data(self):
        """Load and prepare data"""
        print(f"\n[STEP 1] Loading data from: {self.csv_path}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(self.df)} trades")

        # Check features exist
        missing = [col for col in self.feature_cols if col not in self.df.columns]
        if missing:
            print(f"⚠ Missing features: {missing}")
            self.feature_cols = [col for col in self.feature_cols if col in self.df.columns]

        print(f"✓ Features available: {len(self.feature_cols)}")

        # Prepare X and y
        self.X = self.df[self.feature_cols].copy()
        self.y = (self.df[self.target_col] > 0).astype(int)  # 1=win, 0=loss

        # Handle missing values
        self.X = self.X.fillna(0)

        # Check class balance
        win_rate = self.y.mean() * 100
        print(f"✓ Dataset: {len(self.X)} samples, {len(self.feature_cols)} features")
        print(f"✓ Win rate: {win_rate:.1f}%")
        print(f"✓ Sample/Feature ratio: {len(self.X)/len(self.feature_cols):.1f}:1")

        return self.X, self.y

    def correlation_analysis(self, threshold: float = 0.85):
        """
        Step 1: Correlation Analysis

        Remove features with correlation > threshold
        """
        print(f"\n[STEP 2] Correlation Analysis (threshold={threshold})")
        print("-" * 80)

        # Calculate correlation matrix
        corr_matrix = self.X.corr().abs()

        # Find highly correlated pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = []
        high_corr_pairs = []

        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > threshold]
            if len(high_corr) > 0:
                for idx in high_corr.index:
                    high_corr_pairs.append((column, idx, high_corr[idx]))
                    if column not in to_drop:
                        to_drop.append(column)

        print(f"✓ High correlation pairs (>{threshold}):")
        if high_corr_pairs:
            for feat1, feat2, corr_val in high_corr_pairs:
                print(f"   • {feat1} ↔ {feat2}: {corr_val:.3f}")
            print(f"\n⚠ Features to remove: {to_drop}")
        else:
            print("   • None found - all features have low correlation ✓")

        self.results['correlation'] = {
            'high_corr_pairs': high_corr_pairs,
            'features_to_drop': to_drop,
            'corr_matrix': corr_matrix
        }

        return to_drop

    def vif_analysis(self, threshold: float = 10.0):
        """
        Step 2: VIF Analysis (Variance Inflation Factor)

        Detect multicollinearity: VIF > threshold
        """
        print(f"\n[STEP 3] VIF Analysis (threshold={threshold})")
        print("-" * 80)

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data['Feature'] = self.X.columns
        vif_data['VIF'] = [
            variance_inflation_factor(self.X.values, i)
            for i in range(len(self.X.columns))
        ]
        vif_data = vif_data.sort_values('VIF', ascending=False)

        # Find high VIF features
        high_vif = vif_data[vif_data['VIF'] > threshold]

        print("✓ VIF Scores:")
        print(vif_data.to_string(index=False))

        if len(high_vif) > 0:
            print(f"\n⚠ High VIF features (>{threshold}):")
            print(high_vif.to_string(index=False))
        else:
            print(f"\n✓ No high VIF features - multicollinearity OK")

        self.results['vif'] = {
            'vif_scores': vif_data,
            'high_vif_features': high_vif['Feature'].tolist()
        }

        return high_vif['Feature'].tolist()

    def walk_forward_validation(self, n_splits: int = 5):
        """
        Step 3: Walk-Forward Validation

        Test feature stability over time using TimeSeriesSplit
        """
        print(f"\n[STEP 4] Walk-Forward Validation ({n_splits} splits)")
        print("-" * 80)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.X), 1):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            # Train XGBoost
            model = xgb.XGBClassifier(
                max_depth=3,
                n_estimators=50,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train, verbose=False)

            # Evaluate
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5

            fold_results.append({
                'fold': fold,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'accuracy': acc,
                'auc': auc
            })

            print(f"   Fold {fold}: Acc={acc:.3f}, AUC={auc:.3f}, "
                  f"Train={len(train_idx)}, Test={len(test_idx)}")

        # Summary
        fold_df = pd.DataFrame(fold_results)
        avg_acc = fold_df['accuracy'].mean()
        avg_auc = fold_df['auc'].mean()
        std_acc = fold_df['accuracy'].std()

        print(f"\n✓ Average Accuracy: {avg_acc:.3f} ± {std_acc:.3f}")
        print(f"✓ Average AUC: {avg_auc:.3f}")

        self.results['walk_forward'] = {
            'fold_results': fold_df,
            'avg_accuracy': avg_acc,
            'avg_auc': avg_auc,
            'std_accuracy': std_acc
        }

        return fold_df

    def permutation_importance_analysis(self, n_repeats: int = 10):
        """
        Step 4: Permutation Importance

        Shuffle each feature and measure performance drop
        """
        print(f"\n[STEP 5] Permutation Importance ({n_repeats} repeats)")
        print("-" * 80)

        # Train final model on 80% data
        split_idx = int(len(self.X) * 0.8)
        X_train, X_test = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
        y_train, y_test = self.y.iloc[:split_idx], self.y.iloc[split_idx:]

        model = xgb.XGBClassifier(
            max_depth=3,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train, verbose=False)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring='accuracy'
        )

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=False)

        print("✓ Permutation Importance Ranking:")
        print(importance_df.to_string(index=False))

        self.results['permutation_importance'] = {
            'importance_df': importance_df,
            'model': model
        }

        return importance_df

    def ablation_study(self):
        """
        Step 5: Ablation Study

        Remove each feature one at a time and measure performance drop
        """
        print(f"\n[STEP 6] Ablation Study")
        print("-" * 80)

        # Train baseline model (all features)
        split_idx = int(len(self.X) * 0.8)
        X_train, X_test = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
        y_train, y_test = self.y.iloc[:split_idx], self.y.iloc[split_idx:]

        # Baseline
        model_baseline = xgb.XGBClassifier(
            max_depth=3, n_estimators=50, learning_rate=0.1,
            random_state=42, eval_metric='logloss'
        )
        model_baseline.fit(X_train, y_train, verbose=False)
        baseline_acc = accuracy_score(y_test, model_baseline.predict(X_test))

        print(f"✓ Baseline Accuracy (all features): {baseline_acc:.3f}")

        # Test removing each feature
        ablation_results = []

        for feature in self.X.columns:
            # Remove feature
            features_subset = [f for f in self.X.columns if f != feature]
            X_train_sub = X_train[features_subset]
            X_test_sub = X_test[features_subset]

            # Train model without this feature
            model = xgb.XGBClassifier(
                max_depth=3, n_estimators=50, learning_rate=0.1,
                random_state=42, eval_metric='logloss'
            )
            model.fit(X_train_sub, y_train, verbose=False)
            acc = accuracy_score(y_test, model.predict(X_test_sub))

            # Performance drop
            drop = baseline_acc - acc

            ablation_results.append({
                'Feature': feature,
                'Accuracy_without': acc,
                'Performance_drop': drop
            })

        ablation_df = pd.DataFrame(ablation_results).sort_values(
            'Performance_drop', ascending=False
        )

        print("\n✓ Ablation Study Results:")
        print("   (Higher drop = more important feature)")
        print(ablation_df.to_string(index=False))

        self.results['ablation'] = {
            'baseline_accuracy': baseline_acc,
            'ablation_df': ablation_df
        }

        return ablation_df

    def shap_analysis(self):
        """
        Step 6: SHAP Values Analysis

        Explain feature contributions to predictions
        """
        print(f"\n[STEP 7] SHAP Values Analysis")
        print("-" * 80)

        # Use model from permutation importance
        model = self.results['permutation_importance']['model']

        # Calculate SHAP values
        split_idx = int(len(self.X) * 0.8)
        X_test = self.X.iloc[split_idx:]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Mean absolute SHAP values
        shap_importance = pd.DataFrame({
            'Feature': self.X.columns,
            'SHAP_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('SHAP_importance', ascending=False)

        print("✓ SHAP Importance Ranking:")
        print(shap_importance.to_string(index=False))

        self.results['shap'] = {
            'shap_importance': shap_importance,
            'shap_values': shap_values
        }

        return shap_importance

    def generate_final_ranking(self):
        """
        Combine all analyses into final ranking with Keep/Remove recommendations
        """
        print(f"\n[STEP 8] Generating Final Feature Ranking")
        print("=" * 80)

        # Get rankings from each method
        corr_drop = set(self.results['correlation']['features_to_drop'])
        vif_drop = set(self.results['vif']['high_vif_features'])

        perm_ranking = self.results['permutation_importance']['importance_df'].copy()
        perm_ranking['Perm_rank'] = range(1, len(perm_ranking) + 1)

        ablation_ranking = self.results['ablation']['ablation_df'].copy()
        ablation_ranking['Ablation_rank'] = range(1, len(ablation_ranking) + 1)

        shap_ranking = self.results['shap']['shap_importance'].copy()
        shap_ranking['SHAP_rank'] = range(1, len(shap_ranking) + 1)

        # Merge all rankings
        final_df = perm_ranking[['Feature', 'Importance', 'Perm_rank']].merge(
            ablation_ranking[['Feature', 'Performance_drop', 'Ablation_rank']],
            on='Feature'
        ).merge(
            shap_ranking[['Feature', 'SHAP_importance', 'SHAP_rank']],
            on='Feature'
        )

        # Calculate average rank
        final_df['Avg_rank'] = (
            final_df['Perm_rank'] +
            final_df['Ablation_rank'] +
            final_df['SHAP_rank']
        ) / 3

        # Add flags
        final_df['High_correlation'] = final_df['Feature'].isin(corr_drop)
        final_df['High_VIF'] = final_df['Feature'].isin(vif_drop)

        # Recommendation
        def get_recommendation(row):
            if row['High_correlation'] or row['High_VIF']:
                return 'REMOVE'
            elif row['Avg_rank'] <= 8:  # Top 8 features
                return 'KEEP'
            else:
                return 'OPTIONAL'

        final_df['Recommendation'] = final_df.apply(get_recommendation, axis=1)

        # Sort by average rank
        final_df = final_df.sort_values('Avg_rank')

        # Print summary
        print("\n✅ FINAL FEATURE RANKING:")
        print(final_df.to_string(index=False))

        print(f"\n📊 SUMMARY:")
        print(f"   KEEP: {len(final_df[final_df['Recommendation']=='KEEP'])} features")
        print(f"   REMOVE: {len(final_df[final_df['Recommendation']=='REMOVE'])} features")
        print(f"   OPTIONAL: {len(final_df[final_df['Recommendation']=='OPTIONAL'])} features")

        keep_features = final_df[final_df['Recommendation']=='KEEP']['Feature'].tolist()
        print(f"\n✅ RECOMMENDED FEATURES TO KEEP:")
        for i, feat in enumerate(keep_features, 1):
            print(f"   {i}. {feat}")

        self.results['final_ranking'] = final_df

        return final_df

    def run(self):
        """Run full pipeline"""
        try:
            # Load data
            self.load_data()

            # Run analyses
            self.correlation_analysis()
            self.vif_analysis()
            self.walk_forward_validation()
            self.permutation_importance_analysis()
            self.ablation_study()
            self.shap_analysis()

            # Generate final ranking
            final_ranking = self.generate_final_ranking()

            print("\n" + "=" * 80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)

            return final_ranking

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_report(self, output_path: str = 'output/feature_ranking.csv'):
        """Save final ranking to CSV"""
        if 'final_ranking' not in self.results:
            print("⚠ No results to save. Run pipeline first!")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.results['final_ranking'].to_csv(output_path, index=False)
        print(f"\n💾 Report saved to: {output_path}")


def main():
    """
    Main function - Run feature optimization pipeline

    Usage:
        python tools/feature_optimization_pipeline.py
    """
    import argparse

    parser = argparse.ArgumentParser(description='Feature Optimization Pipeline')
    parser.add_argument('--csv', type=str,
                       default='data/backtest_GBPUSD_M15_20251026_162732.csv',
                       help='Path to backtest CSV file')
    parser.add_argument('--output', type=str,
                       default='output/feature_ranking.csv',
                       help='Output path for ranking CSV')

    args = parser.parse_args()

    # Run pipeline
    pipeline = FeatureOptimizationPipeline(args.csv)
    results = pipeline.run()

    if results is not None:
        pipeline.save_report(args.output)

    return pipeline


if __name__ == '__main__':
    main()
