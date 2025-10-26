# core/microstructure/volume_exhaustion.py
"""
Volume Exhaustion Analyzer - Phát hiện price exhaustion qua volume analysis

Core Concept:
- Strong moves start with HIGH volume
- Exhaustion = price moving but volume DECLINING
- Volume divergence = early warning of reversal

Methods:
1. Volume Trend Analysis (volume ma decreasing while price trending)
2. Price-Volume Divergence Detection
3. Climax Volume Detection (blow-off tops/bottoms)
4. Exhaustion Score (0-1)

Use Case:
Price di xa FVG → Volume giam dan → Exhaustion → Ready to return to FVG
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats


class VolumeExhaustionAnalyzer:
    """
    Phát hiện price exhaustion thông qua volume analysis

    Attributes:
        volume_ma_period: Period for volume moving average
        divergence_lookback: Lookback for divergence detection
        climax_threshold: Threshold for climax volume (std multiplier)
    """

    def __init__(self, volume_ma_period: int = 20,
                 divergence_lookback: int = 14,
                 climax_threshold: float = 2.0):
        """
        Initialize Volume Exhaustion Analyzer

        Args:
            volume_ma_period: MA period for volume trend
            divergence_lookback: Lookback window for divergence
            climax_threshold: Std multiplier for climax detection
        """
        self.volume_ma_period = volume_ma_period
        self.divergence_lookback = divergence_lookback
        self.climax_threshold = climax_threshold

    def calculate_volume_trend(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate volume trend (increasing/decreasing)

        Returns positive for increasing, negative for decreasing

        Args:
            data: DataFrame with 'volume'

        Returns:
            pd.Series: Volume trend (-1 to 1)
        """
        volume = data['volume']

        # Calculate volume MA
        volume_ma = volume.rolling(window=self.volume_ma_period, min_periods=5).mean()

        # Calculate slope of volume MA (trend)
        volume_trend = volume_ma.diff(5) / (volume_ma + 1e-8)

        # Normalize to -1 to 1
        volume_trend = np.tanh(volume_trend * 10)

        return volume_trend

    def detect_price_volume_divergence(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect price-volume divergence

        Divergence Types:
        - Bullish: Price declining + Volume declining = exhaustion of selling
        - Bearish: Price rising + Volume declining = exhaustion of buying

        Args:
            data: DataFrame with 'close', 'volume'

        Returns:
            pd.Series: Divergence signals (1=bullish, -1=bearish, 0=none)
        """
        close = data['close']
        volume = data['volume']

        # Calculate trends
        price_trend = close.diff(self.divergence_lookback)
        volume_ma = volume.rolling(window=self.volume_ma_period, min_periods=5).mean()
        volume_trend = volume_ma.diff(self.divergence_lookback)

        divergence = pd.Series(0, index=data.index)

        for i in range(self.divergence_lookback, len(data)):
            p_trend = price_trend.iloc[i]
            v_trend = volume_trend.iloc[i]

            # Bullish divergence: Price down + Volume down
            if p_trend < 0 and v_trend < 0:
                divergence.iloc[i] = 1

            # Bearish divergence: Price up + Volume down
            elif p_trend > 0 and v_trend < 0:
                divergence.iloc[i] = -1

        return divergence

    def detect_climax_volume(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect climax volume (blow-off tops/bottoms)

        Climax = extreme volume spike often marking exhaustion point

        Args:
            data: DataFrame with 'volume', 'close'

        Returns:
            Tuple of (climax_signals, climax_strength)
            climax_signals: 1=bullish climax, -1=bearish climax, 0=none
            climax_strength: Strength of climax (0-1)
        """
        volume = data['volume']
        close = data['close']

        # Calculate volume statistics
        volume_ma = volume.rolling(window=self.volume_ma_period, min_periods=5).mean()
        volume_std = volume.rolling(window=self.volume_ma_period, min_periods=5).std()

        # Z-score of volume
        volume_zscore = (volume - volume_ma) / (volume_std + 1e-8)

        # Detect climax (volume > threshold)
        is_climax = volume_zscore > self.climax_threshold

        # Determine direction
        price_change = close.pct_change()

        climax_signals = pd.Series(0, index=data.index)
        climax_strength = pd.Series(0.0, index=data.index)

        for i in range(len(data)):
            if is_climax.iloc[i]:
                # Bullish climax (panic selling)
                if price_change.iloc[i] < 0:
                    climax_signals.iloc[i] = 1
                # Bearish climax (euphoric buying)
                elif price_change.iloc[i] > 0:
                    climax_signals.iloc[i] = -1

                # Strength based on z-score
                climax_strength.iloc[i] = min(1.0, volume_zscore.iloc[i] / 5.0)

        return climax_signals, climax_strength

    def calculate_exhaustion_score(self, data: pd.DataFrame, index: int,
                                   trend_direction: Optional[str] = None) -> Dict:
        """
        Calculate comprehensive exhaustion score at given index

        Score components:
        1. Volume trend (30%)
        2. Price-volume divergence (40%)
        3. Climax volume (30%)

        Args:
            data: DataFrame with OHLCV
            index: Current index
            trend_direction: 'UP' or 'DOWN' (optional, for directional analysis)

        Returns:
            dict: {
                'exhaustion_score': float (0-1),
                'volume_trend': float,
                'divergence_signal': int,
                'climax_signal': int,
                'is_exhausted': bool,
                'direction': str ('BULLISH' or 'BEARISH')
            }
        """
        if index < self.volume_ma_period:
            return {
                'exhaustion_score': 0.0,
                'volume_trend': 0.0,
                'divergence_signal': 0,
                'climax_signal': 0,
                'is_exhausted': False,
                'direction': 'NEUTRAL'
            }

        # Calculate components
        volume_trend = self.calculate_volume_trend(data)
        divergence_signal = self.detect_price_volume_divergence(data)
        climax_signals, climax_strength = self.detect_climax_volume(data)

        # Get current values
        v_trend = volume_trend.iloc[index]
        div_signal = divergence_signal.iloc[index]
        climax_signal = climax_signals.iloc[index]
        climax_str = climax_strength.iloc[index]

        # Calculate score components

        # Component 1: Volume trend (declining volume = exhaustion)
        volume_score = 0.0
        if v_trend < -0.2:  # Volume declining
            volume_score = abs(v_trend) * 0.3  # Max 0.3

        # Component 2: Price-volume divergence (strongest signal)
        divergence_score = 0.0
        if div_signal != 0:
            divergence_score = 0.4  # Max 0.4

        # Component 3: Climax volume
        climax_score = climax_str * 0.3  # Max 0.3

        # Total exhaustion score
        exhaustion_score = volume_score + divergence_score + climax_score

        # Determine direction
        if div_signal == 1 or (climax_signal == 1 and exhaustion_score > 0.5):
            direction = 'BULLISH'  # Selling exhaustion
        elif div_signal == -1 or (climax_signal == -1 and exhaustion_score > 0.5):
            direction = 'BEARISH'  # Buying exhaustion
        else:
            direction = 'NEUTRAL'

        return {
            'exhaustion_score': exhaustion_score,
            'volume_trend': v_trend,
            'divergence_signal': int(div_signal),
            'climax_signal': int(climax_signal),
            'is_exhausted': exhaustion_score > 0.6,
            'direction': direction,
            'components': {
                'volume_score': volume_score,
                'divergence_score': divergence_score,
                'climax_score': climax_score
            }
        }

    def analyze_exhaustion_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze exhaustion for entire series

        Args:
            data: DataFrame with OHLCV

        Returns:
            DataFrame with exhaustion analysis columns
        """
        result = data.copy()

        # Calculate all components
        result['volume_trend'] = self.calculate_volume_trend(data)
        result['pv_divergence'] = self.detect_price_volume_divergence(data)

        climax_signals, climax_strength = self.detect_climax_volume(data)
        result['climax_signal'] = climax_signals
        result['climax_strength'] = climax_strength

        # Calculate exhaustion scores
        exhaustion_scores = []
        exhaustion_directions = []

        for i in range(len(data)):
            analysis = self.calculate_exhaustion_score(data, i)
            exhaustion_scores.append(analysis['exhaustion_score'])
            exhaustion_directions.append(analysis['direction'])

        result['exhaustion_score'] = exhaustion_scores
        result['exhaustion_direction'] = exhaustion_directions
        result['is_exhausted'] = result['exhaustion_score'] > 0.6

        return result

    def get_exhaustion_zones(self, data: pd.DataFrame,
                            min_score: float = 0.6) -> pd.DataFrame:
        """
        Get zones where exhaustion occurred

        Args:
            data: DataFrame with OHLCV
            min_score: Minimum exhaustion score

        Returns:
            DataFrame with exhaustion zones
        """
        analysis = self.analyze_exhaustion_series(data)

        # Filter exhaustion zones
        exhaustion_zones = analysis[analysis['exhaustion_score'] >= min_score].copy()

        # Add metadata
        exhaustion_zones['zone_type'] = exhaustion_zones['exhaustion_direction']
        exhaustion_zones['zone_strength'] = exhaustion_zones['exhaustion_score']

        return exhaustion_zones[['close', 'volume', 'exhaustion_score',
                                'exhaustion_direction', 'zone_type', 'zone_strength']]
