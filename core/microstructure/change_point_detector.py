# core/microstructure/change_point_detector.py
"""
Change Point Detection - Phát hiện điểm đảo chiều cấu trúc

Methods:
1. CUSUM (Cumulative Sum) - Fast, online detection
2. Bayesian Change Point Detection - More robust
3. Z-Score based detection - Simple but effective

Không cần fixed parameters - adaptive theo volatility
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
from collections import deque


class ChangePointDetector:
    """
    Phát hiện Change Points trong price series

    Change Point = điểm mà statistical properties thay đổi đáng kể
    (mean, variance, trend direction)

    Attributes:
        method: 'cusum', 'bayesian', 'zscore'
        sensitivity: 1.0-3.0 (lower = more sensitive)
        min_distance: Minimum candles between change points
    """

    def __init__(self, method: str = 'cusum', sensitivity: float = 2.0,
                 min_distance: int = 10):
        """
        Initialize Change Point Detector

        Args:
            method: Detection method ('cusum', 'bayesian', 'zscore')
            sensitivity: Detection threshold (1.0=very sensitive, 3.0=conservative)
            min_distance: Min candles between detected points
        """
        self.method = method
        self.sensitivity = sensitivity
        self.min_distance = min_distance

        # State for online detection
        self.reset()

    def reset(self):
        """Reset internal state"""
        self._cumsum_pos = 0.0
        self._cumsum_neg = 0.0
        self._last_change_point = -999
        self._mean_estimate = None
        self._std_estimate = None
        self._history = deque(maxlen=100)  # Keep recent history

    def detect_cusum(self, prices: pd.Series, returns: Optional[pd.Series] = None) -> List[int]:
        """
        CUSUM Change Point Detection

        Phát hiện khi cumulative sum của deviations vượt threshold
        → Indicator của trend change

        Args:
            prices: Price series
            returns: Optional pre-calculated returns

        Returns:
            List[int]: Indices of change points
        """
        if returns is None:
            returns = prices.pct_change().fillna(0)

        # Adaptive threshold based on volatility
        rolling_std = returns.rolling(window=20, min_periods=5).std()
        threshold = self.sensitivity * rolling_std

        change_points = []
        cumsum_pos = 0.0
        cumsum_neg = 0.0

        for i in range(len(returns)):
            ret = returns.iloc[i]

            # Update CUSUM
            cumsum_pos = max(0, cumsum_pos + ret)
            cumsum_neg = min(0, cumsum_neg + ret)

            thresh = threshold.iloc[i] if not pd.isna(threshold.iloc[i]) else 0.02

            # Detect change point
            if cumsum_pos > thresh or cumsum_neg < -thresh:
                # Check minimum distance
                if not change_points or (i - change_points[-1]) >= self.min_distance:
                    change_points.append(i)
                    # Reset CUSUM after detection
                    cumsum_pos = 0.0
                    cumsum_neg = 0.0

        return change_points

    def detect_bayesian(self, prices: pd.Series, window: int = 30) -> List[int]:
        """
        Bayesian Change Point Detection

        Sử dụng Bayesian inference để detect changes in mean/variance
        Robust hơn CUSUM nhưng chậm hơn

        Args:
            prices: Price series
            window: Rolling window size

        Returns:
            List[int]: Indices of change points
        """
        change_points = []

        if len(prices) < window:
            return change_points

        # Calculate rolling statistics
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()

        for i in range(window, len(prices)):
            if i < window * 2:
                continue

            # Compare two windows: before and after
            window1 = prices.iloc[i-window*2:i-window]
            window2 = prices.iloc[i-window:i]

            # T-test for mean difference
            if len(window1) > 5 and len(window2) > 5:
                t_stat, p_value = stats.ttest_ind(window1, window2)

                # F-test for variance difference
                f_stat = np.var(window2, ddof=1) / np.var(window1, ddof=1)

                # Detect significant change
                threshold = 0.05 * self.sensitivity  # p-value threshold

                if p_value < threshold or f_stat > (2 / self.sensitivity) or f_stat < (self.sensitivity / 2):
                    if not change_points or (i - change_points[-1]) >= self.min_distance:
                        change_points.append(i)

        return change_points

    def detect_zscore(self, prices: pd.Series, window: int = 20) -> List[int]:
        """
        Z-Score Based Change Point Detection

        Phát hiện khi price deviation vượt Z-score threshold
        Simple but effective

        Args:
            prices: Price series
            window: Rolling window for mean/std calculation

        Returns:
            List[int]: Indices of change points
        """
        change_points = []

        # Calculate z-scores
        rolling_mean = prices.rolling(window=window, min_periods=5).mean()
        rolling_std = prices.rolling(window=window, min_periods=5).std()

        z_scores = (prices - rolling_mean) / (rolling_std + 1e-8)

        # Detect extreme z-scores (potential reversals)
        threshold = self.sensitivity

        for i in range(window, len(z_scores)):
            z = abs(z_scores.iloc[i])

            if z > threshold:
                if not change_points or (i - change_points[-1]) >= self.min_distance:
                    change_points.append(i)

        return change_points

    def detect(self, prices: pd.Series, **kwargs) -> List[int]:
        """
        Main detection method - routes to specific detector

        Args:
            prices: Price series
            **kwargs: Additional arguments for specific methods

        Returns:
            List[int]: Indices of change points
        """
        if self.method == 'cusum':
            return self.detect_cusum(prices, **kwargs)
        elif self.method == 'bayesian':
            return self.detect_bayesian(prices, **kwargs)
        elif self.method == 'zscore':
            return self.detect_zscore(prices, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def detect_online(self, price: float, timestamp: pd.Timestamp) -> bool:
        """
        Online change point detection (for real-time trading)

        Args:
            price: Current price
            timestamp: Current timestamp

        Returns:
            bool: True if change point detected
        """
        self._history.append(price)

        if len(self._history) < 20:
            return False

        # Calculate rolling statistics
        prices_array = np.array(self._history)
        current_mean = np.mean(prices_array[-20:])
        current_std = np.std(prices_array[-20:])

        # Initialize estimates
        if self._mean_estimate is None:
            self._mean_estimate = current_mean
            self._std_estimate = current_std
            return False

        # CUSUM detection
        deviation = (price - self._mean_estimate) / (self._std_estimate + 1e-8)

        self._cumsum_pos = max(0, self._cumsum_pos + deviation)
        self._cumsum_neg = min(0, self._cumsum_neg + deviation)

        threshold = self.sensitivity

        # Detect change
        if abs(self._cumsum_pos) > threshold or abs(self._cumsum_neg) > threshold:
            # Update estimates
            self._mean_estimate = current_mean
            self._std_estimate = current_std
            self._cumsum_pos = 0.0
            self._cumsum_neg = 0.0
            return True

        # Gradually update estimates
        alpha = 0.1
        self._mean_estimate = alpha * current_mean + (1 - alpha) * self._mean_estimate
        self._std_estimate = alpha * current_std + (1 - alpha) * self._std_estimate

        return False

    def get_change_point_signals(self, prices: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Get change point signals as Series (for integration with strategy)

        Args:
            prices: Price series (typically 'close')
            data: Full OHLCV dataframe

        Returns:
            pd.Series: 1 for bullish change, -1 for bearish change, 0 otherwise
        """
        change_points = self.detect(prices)

        signals = pd.Series(0, index=data.index)

        for cp_idx in change_points:
            if cp_idx < 5 or cp_idx >= len(data):
                continue

            # Determine direction of change
            price_before = prices.iloc[cp_idx - 5:cp_idx].mean()
            price_after = prices.iloc[cp_idx:cp_idx + 5].mean() if cp_idx + 5 < len(prices) else prices.iloc[cp_idx]

            if price_after > price_before:
                signals.iloc[cp_idx] = 1  # Bullish change point
            else:
                signals.iloc[cp_idx] = -1  # Bearish change point

        return signals

    def calculate_change_point_strength(self, prices: pd.Series, cp_index: int,
                                       window: int = 10) -> float:
        """
        Calculate strength of change point (0-1)

        Higher = more significant structural change

        Args:
            prices: Price series
            cp_index: Index of change point
            window: Window to compare before/after

        Returns:
            float: Strength score (0-1)
        """
        if cp_index < window or cp_index + window >= len(prices):
            return 0.0

        # Get before and after windows
        before = prices.iloc[cp_index - window:cp_index]
        after = prices.iloc[cp_index:cp_index + window]

        # Calculate statistics
        mean_diff = abs(after.mean() - before.mean())
        std_diff = abs(after.std() - before.std())

        # Normalize by average std
        avg_std = (before.std() + after.std()) / 2
        if avg_std == 0:
            return 0.0

        strength = (mean_diff / avg_std) + (std_diff / avg_std)

        # Clip to 0-1
        strength = min(1.0, strength / 4.0)

        return strength
