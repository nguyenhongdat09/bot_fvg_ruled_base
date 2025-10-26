# core/microstructure/statistical_swings.py
"""
Statistical Swing Detection - Tìm swing highs/lows với statistical significance

Khác biệt với zigzag thông thường:
- Zigzag: Fixed percentage threshold → overfitting
- Statistical Swing: Adaptive threshold based on volatility → robust

Methods:
1. Fractal-based swing detection (Bill Williams)
2. Statistical significance testing (Z-score)
3. Volume-confirmed swings
4. Adaptive threshold based on ATR

Use Case:
FVG được tạo → Price di xa → Statistical swing high/low → Entry point
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """
    Swing Point Data Structure

    Attributes:
        index: Index in dataframe
        timestamp: Timestamp
        price: Price level
        swing_type: 'HIGH' or 'LOW'
        strength: Statistical significance (0-1)
        volume_confirmed: Volume confirmation
        atr_distance: Distance in ATR units
    """
    index: int
    timestamp: pd.Timestamp
    price: float
    swing_type: str  # 'HIGH' or 'LOW'
    strength: float
    volume_confirmed: bool
    atr_distance: float


class StatisticalSwingDetector:
    """
    Statistical Swing Point Detector

    Attributes:
        fractal_period: Period for fractal detection (default 5)
        zscore_threshold: Z-score threshold for significance (default 1.5)
        volume_confirmation: Require volume confirmation
        min_atr_distance: Minimum distance in ATR units
    """

    def __init__(self, fractal_period: int = 5,
                 zscore_threshold: float = 1.5,
                 volume_confirmation: bool = True,
                 min_atr_distance: float = 1.0):
        """
        Initialize Statistical Swing Detector

        Args:
            fractal_period: Period for fractal detection
            zscore_threshold: Z-score threshold for significance
            volume_confirmation: Require volume spike for confirmation
            min_atr_distance: Minimum swing distance in ATR units
        """
        self.fractal_period = fractal_period
        self.zscore_threshold = zscore_threshold
        self.volume_confirmation = volume_confirmation
        self.min_atr_distance = min_atr_distance

    def detect_fractals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Detect fractal swing points (Bill Williams method)

        Fractal High: Middle candle high > all surrounding highs
        Fractal Low: Middle candle low < all surrounding lows

        Args:
            data: DataFrame with 'high', 'low'

        Returns:
            Tuple of (fractal_highs, fractal_lows)
            Series of boolean values
        """
        high = data['high']
        low = data['low']

        window = self.fractal_period // 2  # Window on each side

        fractal_highs = pd.Series(False, index=data.index)
        fractal_lows = pd.Series(False, index=data.index)

        for i in range(window, len(data) - window):
            # Fractal High
            is_high = True
            center_high = high.iloc[i]

            for j in range(i - window, i + window + 1):
                if j != i and high.iloc[j] >= center_high:
                    is_high = False
                    break

            fractal_highs.iloc[i] = is_high

            # Fractal Low
            is_low = True
            center_low = low.iloc[i]

            for j in range(i - window, i + window + 1):
                if j != i and low.iloc[j] <= center_low:
                    is_low = False
                    break

            fractal_lows.iloc[i] = is_low

        return fractal_highs, fractal_lows

    def calculate_statistical_significance(self, data: pd.DataFrame,
                                          swing_indices: pd.Series,
                                          swing_type: str) -> pd.Series:
        """
        Calculate statistical significance of swing points using Z-score

        Args:
            data: DataFrame with 'high', 'low', 'close'
            swing_indices: Boolean series of swing candidates
            swing_type: 'HIGH' or 'LOW'

        Returns:
            pd.Series: Z-scores for each swing point
        """
        if swing_type == 'HIGH':
            prices = data['high']
        else:
            prices = data['low']

        # Rolling mean and std
        rolling_mean = prices.rolling(window=20, min_periods=5).mean()
        rolling_std = prices.rolling(window=20, min_periods=5).std()

        # Z-scores
        zscores = (prices - rolling_mean) / (rolling_std + 1e-8)

        # Only keep z-scores for swing points
        result = pd.Series(0.0, index=data.index)
        result[swing_indices] = abs(zscores[swing_indices])

        return result

    def check_volume_confirmation(self, data: pd.DataFrame,
                                  swing_indices: pd.Series) -> pd.Series:
        """
        Check if swing points are confirmed by volume spike

        Args:
            data: DataFrame with 'volume'
            swing_indices: Boolean series of swing candidates

        Returns:
            pd.Series: Boolean series of volume-confirmed swings
        """
        volume = data['volume']

        # Calculate volume z-score
        volume_ma = volume.rolling(window=20, min_periods=5).mean()
        volume_std = volume.rolling(window=20, min_periods=5).std()
        volume_zscore = (volume - volume_ma) / (volume_std + 1e-8)

        # Volume spike threshold (1.5 std above mean)
        is_spike = volume_zscore > 1.5

        # Confirm swings with volume spikes
        confirmed = swing_indices & is_spike

        return confirmed

    def calculate_atr_distance(self, data: pd.DataFrame, atr_period: int = 14) -> pd.Series:
        """
        Calculate ATR for distance measurement

        Args:
            data: DataFrame with 'high', 'low', 'close'
            atr_period: ATR period

        Returns:
            pd.Series: ATR values
        """
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()

        return atr

    def detect_swings(self, data: pd.DataFrame, atr_series: Optional[pd.Series] = None) -> List[SwingPoint]:
        """
        Detect all statistical swing points

        Args:
            data: DataFrame with OHLCV
            atr_series: Pre-calculated ATR (optional)

        Returns:
            List[SwingPoint]: List of detected swing points
        """
        # Calculate ATR if not provided
        if atr_series is None:
            atr_series = self.calculate_atr_distance(data)

        # Step 1: Detect fractals
        fractal_highs, fractal_lows = self.detect_fractals(data)

        # Step 2: Calculate statistical significance
        high_significance = self.calculate_statistical_significance(data, fractal_highs, 'HIGH')
        low_significance = self.calculate_statistical_significance(data, fractal_lows, 'LOW')

        # Step 3: Filter by z-score threshold
        significant_highs = (high_significance >= self.zscore_threshold) & fractal_highs
        significant_lows = (low_significance >= self.zscore_threshold) & fractal_lows

        # Step 4: Volume confirmation (if enabled)
        if self.volume_confirmation:
            high_volume_confirmed = self.check_volume_confirmation(data, significant_highs)
            low_volume_confirmed = self.check_volume_confirmation(data, significant_lows)
        else:
            high_volume_confirmed = significant_highs
            low_volume_confirmed = significant_lows

        # Step 5: Build swing points
        swing_points = []

        # Process highs
        for i in range(len(data)):
            if significant_highs.iloc[i]:
                price = data.iloc[i]['high']
                atr = atr_series.iloc[i]

                swing = SwingPoint(
                    index=i,
                    timestamp=data.index[i],
                    price=price,
                    swing_type='HIGH',
                    strength=high_significance.iloc[i] / 5.0,  # Normalize to 0-1
                    volume_confirmed=high_volume_confirmed.iloc[i],
                    atr_distance=atr if not pd.isna(atr) else 0.0
                )
                swing_points.append(swing)

        # Process lows
        for i in range(len(data)):
            if significant_lows.iloc[i]:
                price = data.iloc[i]['low']
                atr = atr_series.iloc[i]

                swing = SwingPoint(
                    index=i,
                    timestamp=data.index[i],
                    price=price,
                    swing_type='LOW',
                    strength=low_significance.iloc[i] / 5.0,
                    volume_confirmed=low_volume_confirmed.iloc[i],
                    atr_distance=atr if not pd.isna(atr) else 0.0
                )
                swing_points.append(swing)

        # Sort by index
        swing_points.sort(key=lambda x: x.index)

        # Filter by minimum ATR distance
        filtered_swings = self._filter_by_atr_distance(swing_points)

        return filtered_swings

    def _filter_by_atr_distance(self, swing_points: List[SwingPoint]) -> List[SwingPoint]:
        """
        Filter swing points by minimum ATR distance

        Args:
            swing_points: List of swing points

        Returns:
            List[SwingPoint]: Filtered list
        """
        if not swing_points:
            return []

        filtered = [swing_points[0]]  # Keep first swing

        for swing in swing_points[1:]:
            last_swing = filtered[-1]

            # Calculate distance in ATR units
            price_diff = abs(swing.price - last_swing.price)
            avg_atr = (swing.atr_distance + last_swing.atr_distance) / 2

            if avg_atr > 0:
                atr_distance = price_diff / avg_atr
            else:
                atr_distance = 0

            # Keep if distance is sufficient
            if atr_distance >= self.min_atr_distance:
                filtered.append(swing)

        return filtered

    def get_recent_swing(self, swing_points: List[SwingPoint],
                        current_index: int,
                        swing_type: Optional[str] = None,
                        max_lookback: int = 50) -> Optional[SwingPoint]:
        """
        Get most recent swing point before current index

        Args:
            swing_points: List of swing points
            current_index: Current index
            swing_type: 'HIGH' or 'LOW' (None = any type)
            max_lookback: Maximum candles to look back

        Returns:
            SwingPoint or None
        """
        min_index = max(0, current_index - max_lookback)

        # Filter swings before current index
        candidates = [
            swing for swing in swing_points
            if swing.index < current_index and swing.index >= min_index
        ]

        # Filter by type if specified
        if swing_type:
            candidates = [swing for swing in candidates if swing.swing_type == swing_type]

        if not candidates:
            return None

        # Return most recent
        return max(candidates, key=lambda x: x.index)

    def detect_swing_at_index(self, data: pd.DataFrame, index: int,
                             atr_series: Optional[pd.Series] = None) -> Optional[SwingPoint]:
        """
        Detect if there's a swing point at specific index (for online detection)

        Args:
            data: DataFrame with OHLCV (up to current index)
            index: Index to check
            atr_series: Pre-calculated ATR (optional)

        Returns:
            SwingPoint or None
        """
        if index < self.fractal_period:
            return None

        # Check if index is a fractal
        window = self.fractal_period // 2

        if index + window >= len(data):
            return None  # Need future candles for confirmation

        # Calculate ATR
        if atr_series is None:
            atr_series = self.calculate_atr_distance(data)

        # Check fractal high
        is_high_fractal = True
        center_high = data.iloc[index]['high']

        for j in range(index - window, index + window + 1):
            if j != index and data.iloc[j]['high'] >= center_high:
                is_high_fractal = False
                break

        # Check fractal low
        is_low_fractal = True
        center_low = data.iloc[index]['low']

        for j in range(index - window, index + window + 1):
            if j != index and data.iloc[j]['low'] <= center_low:
                is_low_fractal = False
                break

        # If fractal found, calculate significance
        if is_high_fractal:
            # Calculate z-score
            prices = data['high'].iloc[:index+1]
            rolling_mean = prices.rolling(window=20, min_periods=5).mean().iloc[-1]
            rolling_std = prices.rolling(window=20, min_periods=5).std().iloc[-1]
            zscore = abs((center_high - rolling_mean) / (rolling_std + 1e-8))

            if zscore >= self.zscore_threshold:
                return SwingPoint(
                    index=index,
                    timestamp=data.index[index],
                    price=center_high,
                    swing_type='HIGH',
                    strength=min(1.0, zscore / 5.0),
                    volume_confirmed=True,  # Simplified for online detection
                    atr_distance=atr_series.iloc[index] if not pd.isna(atr_series.iloc[index]) else 0.0
                )

        elif is_low_fractal:
            # Calculate z-score
            prices = data['low'].iloc[:index+1]
            rolling_mean = prices.rolling(window=20, min_periods=5).mean().iloc[-1]
            rolling_std = prices.rolling(window=20, min_periods=5).std().iloc[-1]
            zscore = abs((center_low - rolling_mean) / (rolling_std + 1e-8))

            if zscore >= self.zscore_threshold:
                return SwingPoint(
                    index=index,
                    timestamp=data.index[index],
                    price=center_low,
                    swing_type='LOW',
                    strength=min(1.0, zscore / 5.0),
                    volume_confirmed=True,
                    atr_distance=atr_series.iloc[index] if not pd.isna(atr_series.iloc[index]) else 0.0
                )

        return None
