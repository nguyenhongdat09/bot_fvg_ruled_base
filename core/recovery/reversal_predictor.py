"""
Reversal Predictor for Smart DCA Recovery System

Predicts probability of price reversal using:
1. RULE-BASED mode: Uses exhaustion indicators (CUSUM, velocity, RSI divergence)
2. ML mode: Uses trained LightGBM model with same features

Supports easy switching between modes via config.

Author: Claude Code
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ReversalSignal:
    """Reversal prediction result"""
    probability: float  # 0.0 to 1.0
    confidence: str     # 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH'
    signals: Dict       # Individual signal contributions
    mode: str          # 'rule-based' or 'ml'

    def __str__(self):
        return f"Reversal: {self.probability*100:.1f}% ({self.confidence}, {self.mode})"


class ReversalPredictor:
    """
    Predict price reversal probability using rule-based OR ML approach

    RULE-BASED MODE (no training required):
    - Uses exhaustion indicators (CUSUM, velocity, acceleration)
    - RSI divergence detection
    - Volume analysis
    - Statistical excursion percentile
    - Combines signals with weighted scoring

    ML MODE (requires training):
    - Uses same features as rule-based
    - Trains LightGBM classifier
    - Learns optimal feature weights from historical data
    - Better at capturing non-linear relationships

    Usage:
        # Rule-based (default, no training needed)
        predictor = ReversalPredictor(mode='rule-based')
        predictor.initialize(data, statistical_analyzer)

        # ML mode (requires training)
        predictor = ReversalPredictor(mode='ml')
        predictor.initialize(data, statistical_analyzer)
        predictor.train(data)  # Train on historical data

        # Predict reversal
        signal = predictor.predict(
            current_idx=1000,
            entry_price=1.2500,
            current_price=1.2600,
            direction='sell'
        )
        print(signal)  # Reversal: 75.3% (HIGH, rule-based)
    """

    def __init__(self, mode: str = 'rule-based'):
        """
        Initialize reversal predictor

        Args:
            mode: 'rule-based' or 'ml'
        """
        if mode not in ['rule-based', 'ml']:
            raise ValueError(f"Mode must be 'rule-based' or 'ml', got: {mode}")

        self.mode = mode
        self.data = None
        self.statistical_analyzer = None
        self.model = None

        # Rule-based weights (tuned for typical forex behavior)
        self.rule_weights = {
            'statistical_excursion': 0.30,  # Highest weight - based on historical percentiles
            'cusum_exhaustion': 0.25,       # CUSUM changepoint detection
            'price_velocity': 0.20,         # Price velocity exhaustion
            'rsi_divergence': 0.15,         # RSI divergence
            'volume_divergence': 0.10,      # Volume divergence
        }

    def initialize(self, data: pd.DataFrame, statistical_analyzer):
        """
        Initialize with data and statistical analyzer

        Args:
            data: OHLC dataframe
            statistical_analyzer: StatisticalAnalyzer instance
        """
        self.data = data.copy()
        self.statistical_analyzer = statistical_analyzer

        # Calculate features
        self._calculate_features()

    def _calculate_features(self):
        """Calculate all features needed for prediction"""

        # 1. Price velocity and acceleration
        self.data['returns'] = self.data['close'].pct_change()
        self.data['velocity'] = self.data['returns'].rolling(5).mean()
        self.data['acceleration'] = self.data['velocity'].diff()

        # 2. RSI
        self.data['rsi'] = self._calculate_rsi(self.data['close'], period=14)

        # 3. Volume (if available, else use tick_volume or set to 1)
        if 'tick_volume' in self.data.columns:
            self.data['volume'] = self.data['tick_volume']
        elif 'volume' not in self.data.columns:
            self.data['volume'] = 1

        # 4. ATR for volatility normalization
        self.data['atr'] = self._calculate_atr(self.data, period=14)

        # 5. Bollinger Bands for extremes
        self.data['bb_upper'], self.data['bb_lower'] = self._calculate_bollinger_bands(
            self.data['close'], period=20, std=2
        )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR"""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()

        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)

        return upper, lower

    def _extract_features(self,
                         idx: int,
                         entry_price: float,
                         current_price: float,
                         direction: str) -> Dict:
        """
        Extract features for prediction at given index

        Args:
            idx: Current index in data
            entry_price: Entry price of position
            current_price: Current market price
            direction: 'buy' or 'sell'

        Returns:
            Dict of features
        """
        if idx < 50:
            raise ValueError("Need at least 50 bars of history")

        # Get current candle
        candle = self.data.iloc[idx]

        # 1. Statistical excursion percentile
        excursion_pips = abs(current_price - entry_price) / 0.0001
        stats = self.statistical_analyzer.analyze_excursions(
            'bear' if direction.lower() == 'buy' else 'bull'
        )

        if excursion_pips <= stats.percentile_50:
            excursion_score = 0.2
        elif excursion_pips <= stats.percentile_75:
            excursion_score = 0.5
        elif excursion_pips <= stats.percentile_90:
            excursion_score = 0.8
        else:
            excursion_score = 0.95

        # 2. CUSUM exhaustion (use existing indicator)
        try:
            from core.indicators.exhaustion_indicators import ExhaustionIndicators
            exhaustion = ExhaustionIndicators(self.data.iloc[:idx+1])
            cusum_score = exhaustion.calculate_exhaustion_score() / 100.0  # Normalize to 0-1
        except:
            # Fallback: Use velocity-based exhaustion
            recent_velocity = abs(self.data['velocity'].iloc[idx-10:idx].mean())
            current_velocity = abs(self.data['velocity'].iloc[idx])
            cusum_score = 0.5 if current_velocity < recent_velocity * 0.5 else 0.2

        # 3. Price velocity exhaustion
        # When velocity slows down significantly = exhaustion
        velocity_window = self.data['velocity'].iloc[idx-20:idx]
        current_velocity = abs(self.data['velocity'].iloc[idx])
        avg_velocity = abs(velocity_window.mean())

        if avg_velocity > 0:
            velocity_ratio = current_velocity / avg_velocity
            velocity_score = 1.0 - min(velocity_ratio, 1.0)  # Lower velocity = higher score
        else:
            velocity_score = 0.5

        # 4. RSI divergence
        # Price makes new extreme but RSI doesn't = divergence
        rsi = self.data['rsi'].iloc[idx]

        if direction.lower() == 'buy':
            # For long position, check if price at new low but RSI not
            price_window = self.data['low'].iloc[idx-20:idx]
            rsi_window = self.data['rsi'].iloc[idx-20:idx]

            if current_price <= price_window.min() and rsi > rsi_window.min():
                rsi_divergence_score = 0.9  # Bullish divergence
            elif rsi < 30:
                rsi_divergence_score = 0.7  # Oversold
            else:
                rsi_divergence_score = 0.3
        else:
            # For short position, check if price at new high but RSI not
            price_window = self.data['high'].iloc[idx-20:idx]
            rsi_window = self.data['rsi'].iloc[idx-20:idx]

            if current_price >= price_window.max() and rsi < rsi_window.max():
                rsi_divergence_score = 0.9  # Bearish divergence
            elif rsi > 70:
                rsi_divergence_score = 0.7  # Overbought
            else:
                rsi_divergence_score = 0.3

        # 5. Volume divergence
        # Price makes extreme but volume decreases = exhaustion
        volume_window = self.data['volume'].iloc[idx-10:idx]
        current_volume = self.data['volume'].iloc[idx]
        avg_volume = volume_window.mean()

        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio < 0.7:  # Volume declining
                volume_divergence_score = 0.8
            elif volume_ratio > 1.3:  # Volume spiking (climax?)
                volume_divergence_score = 0.6
            else:
                volume_divergence_score = 0.3
        else:
            volume_divergence_score = 0.5

        return {
            'statistical_excursion': excursion_score,
            'cusum_exhaustion': cusum_score,
            'price_velocity': velocity_score,
            'rsi_divergence': rsi_divergence_score,
            'volume_divergence': volume_divergence_score,
            # Additional context (not used in scoring but useful for ML)
            'excursion_pips': excursion_pips,
            'rsi': rsi,
            'atr': candle['atr'],
            'current_velocity': current_velocity,
        }

    def predict(self,
                current_idx: int,
                entry_price: float,
                current_price: float,
                direction: str) -> ReversalSignal:
        """
        Predict reversal probability

        Args:
            current_idx: Current index in data
            entry_price: Entry price of position
            current_price: Current market price
            direction: Trade direction ('buy' or 'sell')

        Returns:
            ReversalSignal with probability and details
        """
        if self.data is None:
            raise ValueError("Must call initialize() first")

        # Extract features
        features = self._extract_features(current_idx, entry_price, current_price, direction)

        if self.mode == 'rule-based':
            # Rule-based scoring
            probability = 0.0
            for feature, score in features.items():
                if feature in self.rule_weights:
                    probability += score * self.rule_weights[feature]

            # Ensure in range [0, 1]
            probability = max(0.0, min(1.0, probability))

        else:
            # ML-based prediction
            if self.model is None:
                raise ValueError("ML model not trained. Call train() first or use mode='rule-based'")

            # Prepare features for model
            X = np.array([[
                features['statistical_excursion'],
                features['cusum_exhaustion'],
                features['price_velocity'],
                features['rsi_divergence'],
                features['volume_divergence'],
                features['excursion_pips'] / 100,  # Normalize
                features['rsi'] / 100,
                features['atr'] / features['atr'] if features['atr'] > 0 else 1,  # Normalize
            ]])

            # Predict probability
            probability = self.model.predict_proba(X)[0][1]  # Probability of reversal class

        # Determine confidence level
        if probability >= 0.80:
            confidence = 'VERY_HIGH'
        elif probability >= 0.65:
            confidence = 'HIGH'
        elif probability >= 0.50:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return ReversalSignal(
            probability=probability,
            confidence=confidence,
            signals=features,
            mode=self.mode
        )

    def train(self, train_data: pd.DataFrame = None, test_size: float = 0.2):
        """
        Train ML model (only for ML mode)

        Args:
            train_data: Training data (if None, uses self.data)
            test_size: Fraction of data to use for testing
        """
        if self.mode != 'ml':
            raise ValueError("train() only available in ML mode")

        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, roc_auc_score
        except ImportError:
            raise ImportError(
                "ML mode requires: pip install lightgbm scikit-learn\n"
                "Alternatively, use mode='rule-based' which requires no additional packages"
            )

        if train_data is None:
            train_data = self.data

        print("\n[ML Training] Preparing training data...")

        # Generate training samples
        X_list = []
        y_list = []

        # Analyze both directions
        for direction in ['buy', 'sell']:
            excursion_direction = 'bear' if direction == 'buy' else 'bull'
            stats = self.statistical_analyzer.analyze_excursions(excursion_direction)

            # Simulate entries at swing points
            swing_indices = (self.statistical_analyzer.swing_low_indices if direction == 'buy'
                           else self.statistical_analyzer.swing_high_indices)

            for entry_idx in swing_indices[:int(len(swing_indices) * 0.8)]:  # Use 80% for training
                if entry_idx < 50 or entry_idx > len(train_data) - 50:
                    continue

                entry_price = train_data['close'].iloc[entry_idx]

                # Sample points along the excursion
                for offset in [10, 20, 30, 40]:
                    check_idx = entry_idx + offset
                    if check_idx >= len(train_data):
                        break

                    current_price = train_data['close'].iloc[check_idx]

                    # Extract features
                    try:
                        features = self._extract_features(check_idx, entry_price, current_price, direction)

                        # Label: Did price reverse within next 20 bars?
                        future_prices = train_data['close'].iloc[check_idx:check_idx+20]
                        if direction == 'buy':
                            # Check if price went back up
                            reversed = (future_prices.max() >= entry_price)
                        else:
                            # Check if price went back down
                            reversed = (future_prices.min() <= entry_price)

                        X_list.append([
                            features['statistical_excursion'],
                            features['cusum_exhaustion'],
                            features['price_velocity'],
                            features['rsi_divergence'],
                            features['volume_divergence'],
                            features['excursion_pips'] / 100,
                            features['rsi'] / 100,
                            features['atr'] / features['atr'] if features['atr'] > 0 else 1,
                        ])
                        y_list.append(1 if reversed else 0)
                    except:
                        continue

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Generated {len(X)} training samples")
        print(f"Reversal rate: {y.mean()*100:.1f}%")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train LightGBM
        print("\n[ML Training] Training LightGBM model...")

        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n[ML Training] Model trained successfully!")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  AUC-ROC: {auc:.3f}")

        # Feature importance
        importance = self.model.feature_importances_
        feature_names = ['statistical_excursion', 'cusum_exhaustion', 'price_velocity',
                        'rsi_divergence', 'volume_divergence', 'excursion_pips',
                        'rsi', 'atr']

        print("\n[ML Training] Feature Importance:")
        for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.3f}")


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/home/user/bot_fvg_ruled_base')
    from config import DATA_DIR
    from core.recovery.statistical_analyzer import StatisticalAnalyzer

    print("\n" + "="*80)
    print("REVERSAL PREDICTOR - EXAMPLE")
    print("="*80)

    # Load data
    data_file = DATA_DIR / 'GBPUSD_M15_180days.csv'

    if not data_file.exists():
        print(f"\nError: Data file not found: {data_file}")
        sys.exit(1)

    print(f"\nLoading data from: {data_file}")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} candles")

    # Initialize statistical analyzer
    print("\nInitializing Statistical Analyzer...")
    stat_analyzer = StatisticalAnalyzer(data, pip_value=0.0001)

    # Test both modes
    for mode in ['rule-based', 'ml']:
        print(f"\n{'='*80}")
        print(f"MODE: {mode.upper()}")
        print("="*80)

        predictor = ReversalPredictor(mode=mode)
        predictor.initialize(data, stat_analyzer)

        if mode == 'ml':
            # Train ML model
            predictor.train(data)

        # Test prediction
        print(f"\n[EXAMPLE] Simulating SELL trade @ 1.2700:")
        entry_price = 1.2700
        test_prices = [1.2750, 1.2800, 1.2850, 1.2900]

        for current_price in test_prices:
            # Find index where price is close to current_price
            idx = (data['close'] - current_price).abs().idxmin()
            idx_pos = data.index.get_loc(idx)

            if idx_pos >= 50:
                signal = predictor.predict(idx_pos, entry_price, current_price, 'sell')
                pips = abs(current_price - entry_price) / 0.0001
                print(f"  Price: {current_price:.4f} ({pips:.0f} pips adverse) -> {signal}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80 + "\n")
