"""
Exhaustion Detection Indicators - Advanced Statistical Methods

Solves the "timing problem" in FVG trading:
- FVG detected ✅
- Price pullback to FVG ✅
- **When to enter?** ← This module solves this!

Methods implemented:
1. CUSUM Changepoint Detection - Detect momentum regime changes
2. Price Velocity & Acceleration - Physics-based exhaustion detection

Author: Claude Code
Date: 2025-10-26
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy.signal import savgol_filter


class CUSUMChangepoint:
    """
    CUSUM (Cumulative Sum) Changepoint Detection

    Detects when price momentum regime CHANGES (from pullback → reversal)

    Theory:
    - CUSUM tracks cumulative deviations from expected behavior
    - When CUSUM exceeds threshold → structural break detected!
    - Used in manufacturing quality control, now applied to trading

    Advantages:
    - NO fixed parameters (threshold = 3σ is statistical constant)
    - Adaptive to volatility
    - Academic proven method
    """

    def __init__(self, lookback: int = 20, threshold_sigma: float = 3.0):
        """
        Args:
            lookback: Window for calculating baseline statistics
            threshold_sigma: Number of std deviations for changepoint (default 3.0 = 99.7% confidence)
        """
        self.lookback = lookback
        self.threshold = threshold_sigma

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect changepoints in price momentum

        Returns DataFrame with columns:
        - cusum_pos: Positive CUSUM (upward momentum)
        - cusum_neg: Negative CUSUM (downward momentum)
        - changepoint_score: 0-100 (strength of changepoint signal)
        - changepoint_direction: 'bullish_exhaustion', 'bearish_exhaustion', or 'none'
        """

        close = data['close'].values
        returns = np.diff(close) / (close[:-1] + 1e-8)

        # Initialize output arrays
        n = len(close)
        cusum_pos_arr = np.zeros(n)
        cusum_neg_arr = np.zeros(n)
        changepoint_score_arr = np.zeros(n)
        changepoint_direction_arr = np.array(['none'] * n, dtype=object)

        # Calculate CUSUM for each point
        for i in range(self.lookback, len(returns)):
            window_returns = returns[i-self.lookback:i]

            # Baseline statistics
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns)

            if std_return < 1e-8:
                continue

            # Normalize returns
            normalized = (window_returns - mean_return) / std_return

            # CUSUM calculation
            # Positive CUSUM: Cumulative sum of positive deviations
            cusum_pos = np.maximum.accumulate(
                np.maximum(0, np.cumsum(normalized - 0.5))
            )

            # Negative CUSUM: Cumulative sum of negative deviations
            cusum_neg = np.maximum.accumulate(
                np.maximum(0, np.cumsum(-normalized - 0.5))
            )

            # Get current CUSUM values
            cusum_pos_val = cusum_pos[-1] if len(cusum_pos) > 0 else 0
            cusum_neg_val = cusum_neg[-1] if len(cusum_neg) > 0 else 0

            cusum_pos_arr[i+1] = cusum_pos_val
            cusum_neg_arr[i+1] = cusum_neg_val

            # Detect changepoint
            if cusum_pos_val > self.threshold:
                # Upward momentum exhausted → bearish exhaustion
                changepoint_direction_arr[i+1] = 'bearish_exhaustion'
                changepoint_score_arr[i+1] = min(100, (cusum_pos_val / self.threshold) * 100)

            elif cusum_neg_val > self.threshold:
                # Downward momentum exhausted → bullish exhaustion
                changepoint_direction_arr[i+1] = 'bullish_exhaustion'
                changepoint_score_arr[i+1] = min(100, (cusum_neg_val / self.threshold) * 100)

            else:
                changepoint_direction_arr[i+1] = 'none'
                changepoint_score_arr[i+1] = 0

        # Create result DataFrame
        result = pd.DataFrame({
            'cusum_pos': cusum_pos_arr,
            'cusum_neg': cusum_neg_arr,
            'changepoint_score': changepoint_score_arr,
            'changepoint_direction': changepoint_direction_arr
        }, index=data.index)

        return result


class PriceVelocityAcceleration:
    """
    Price Velocity & Acceleration - Physics-Based Exhaustion Detection

    Treats price as a projectile in motion:
    - Velocity = rate of price change (first derivative)
    - Acceleration = rate of velocity change (second derivative)

    Exhaustion signals:
    - Velocity approaching zero → Price losing momentum
    - Acceleration reversing → Direction about to change

    Advantages:
    - Physics-based (universal laws, no parameters to tune!)
    - Scale-invariant (normalized by ATR)
    - Works on any timeframe/instrument
    """

    def __init__(self, smooth_window: int = 3):
        """
        Args:
            smooth_window: Window for Savitzky-Golay smoothing (must be odd)
                          Reduce noise without lag
        """
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate velocity and acceleration, detect exhaustion

        Returns DataFrame with columns:
        - velocity: Price velocity (normalized by ATR)
        - acceleration: Price acceleration (normalized by ATR)
        - exhaustion_score: 0-100 (strength of exhaustion signal)
        - exhaustion_direction: 'bullish_exhaustion', 'bearish_exhaustion', or 'none'
        """

        close = data['close'].values
        n = len(close)

        # Smooth price using Savitzky-Golay filter (preserves peaks/troughs)
        window_length = min(self.smooth_window * 2 + 1, n)
        if window_length < 5:
            # Not enough data, return zeros
            return pd.DataFrame({
                'velocity': np.zeros(n),
                'acceleration': np.zeros(n),
                'exhaustion_score': np.zeros(n),
                'exhaustion_direction': np.array(['none'] * n, dtype=object)
            }, index=data.index)

        smoothed = savgol_filter(close, window_length=window_length, polyorder=2)

        # Calculate derivatives
        velocity = np.gradient(smoothed)
        acceleration = np.gradient(velocity)

        # Normalize by ATR to be scale-invariant
        if 'ATR' in data.columns:
            atr = data['ATR'].values
        else:
            # Fallback: use rolling std of returns
            returns = np.diff(close) / (close[:-1] + 1e-8)
            atr = np.concatenate([[returns[0]], returns])
            atr = pd.Series(atr).rolling(14, min_periods=1).std().values

        atr = np.where(atr < 1e-8, 1e-8, atr)  # Prevent division by zero

        velocity_norm = velocity / atr
        acceleration_norm = acceleration / atr

        # Detect exhaustion
        exhaustion_score_arr = np.zeros(n)
        exhaustion_direction_arr = np.array(['none'] * n, dtype=object)

        for i in range(n):
            v = velocity_norm[i]
            a = acceleration_norm[i]

            # Bullish exhaustion:
            # - Negative velocity (price falling) approaching zero
            # - Positive acceleration (deceleration of fall)
            if v < 0 and v > -0.5 and a > 0:
                exhaustion_direction_arr[i] = 'bullish_exhaustion'
                # Score based on how close velocity is to zero
                exhaustion_score_arr[i] = 100 * (1 - abs(v) / 0.5)

            # Bearish exhaustion:
            # - Positive velocity (price rising) approaching zero
            # - Negative acceleration (deceleration of rise)
            elif v > 0 and v < 0.5 and a < 0:
                exhaustion_direction_arr[i] = 'bearish_exhaustion'
                exhaustion_score_arr[i] = 100 * (1 - abs(v) / 0.5)

            else:
                exhaustion_direction_arr[i] = 'none'
                exhaustion_score_arr[i] = 0

        # Create result DataFrame
        result = pd.DataFrame({
            'velocity': velocity_norm,
            'acceleration': acceleration_norm,
            'exhaustion_score': exhaustion_score_arr,
            'exhaustion_direction': exhaustion_direction_arr
        }, index=data.index)

        return result


class ExhaustionDetector:
    """
    Combines multiple statistical methods to detect pullback exhaustion

    Uses:
    1. CUSUM Changepoint Detection (35% weight)
    2. Price Velocity & Acceleration (30% weight)

    Confluence approach: Only signal when BOTH methods agree

    Usage in FVG strategy:
    - FVG Bullish detected
    - Wait for bearish pullback
    - When ExhaustionDetector says "bearish_exhaustion" → ENTER LONG!
    """

    def __init__(self,
                 cusum_lookback: int = 20,
                 cusum_threshold: float = 3.0,
                 velocity_smooth_window: int = 3):
        """
        Initialize exhaustion detector with both methods

        Args:
            cusum_lookback: Lookback window for CUSUM
            cusum_threshold: Sigma threshold for changepoint (default 3.0)
            velocity_smooth_window: Smoothing window for velocity calculation
        """
        self.cusum = CUSUMChangepoint(lookback=cusum_lookback, threshold_sigma=cusum_threshold)
        self.velocity = PriceVelocityAcceleration(smooth_window=velocity_smooth_window)

        # Confluence weights
        self.weights = {
            'cusum': 0.55,      # CUSUM is stronger signal
            'velocity': 0.45    # Velocity is confirmatory
        }

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate exhaustion confluence score

        Returns DataFrame with columns:
        - exhaustion_score: 0-100 (combined score from both methods)
        - exhaustion_direction: 'bullish_exhaustion', 'bearish_exhaustion', or 'none'
        - cusum_score: Score from CUSUM method
        - cusum_direction: Direction from CUSUM
        - velocity_score: Score from velocity method
        - velocity_direction: Direction from velocity method
        """

        # Calculate both methods
        cusum_result = self.cusum.detect(data)
        velocity_result = self.velocity.detect(data)

        n = len(data)
        exhaustion_score_arr = np.zeros(n)
        exhaustion_direction_arr = np.array(['none'] * n, dtype=object)

        # Combine scores
        for i in range(n):
            cusum_score = cusum_result.iloc[i]['changepoint_score']
            cusum_dir = cusum_result.iloc[i]['changepoint_direction']

            velocity_score = velocity_result.iloc[i]['exhaustion_score']
            velocity_dir = velocity_result.iloc[i]['exhaustion_direction']

            # Only combine if directions AGREE
            if cusum_dir == velocity_dir and cusum_dir != 'none':
                # Weighted average
                combined_score = (
                    cusum_score * self.weights['cusum'] +
                    velocity_score * self.weights['velocity']
                )
                exhaustion_score_arr[i] = combined_score
                exhaustion_direction_arr[i] = cusum_dir

            # If only one method signals (weaker confidence)
            elif cusum_dir != 'none' and velocity_dir == 'none':
                exhaustion_score_arr[i] = cusum_score * 0.6  # Reduced confidence
                exhaustion_direction_arr[i] = cusum_dir

            elif velocity_dir != 'none' and cusum_dir == 'none':
                exhaustion_score_arr[i] = velocity_score * 0.6
                exhaustion_direction_arr[i] = velocity_dir

            else:
                exhaustion_score_arr[i] = 0
                exhaustion_direction_arr[i] = 'none'

        # Create result DataFrame
        result = pd.DataFrame({
            'exhaustion_score': exhaustion_score_arr,
            'exhaustion_direction': exhaustion_direction_arr,
            'cusum_score': cusum_result['changepoint_score'],
            'cusum_direction': cusum_result['changepoint_direction'],
            'velocity_score': velocity_result['exhaustion_score'],
            'velocity_direction': velocity_result['exhaustion_direction'],
            # Also include raw CUSUM and velocity values for analysis
            'cusum_pos': cusum_result['cusum_pos'],
            'cusum_neg': cusum_result['cusum_neg'],
            'velocity': velocity_result['velocity'],
            'acceleration': velocity_result['acceleration']
        }, index=data.index)

        return result

    @staticmethod
    def calculate_all_exhaustion_indicators(data: pd.DataFrame,
                                            cusum_lookback: int = 20,
                                            cusum_threshold: float = 3.0,
                                            velocity_smooth_window: int = 3) -> pd.DataFrame:
        """
        Static method to calculate all exhaustion indicators and add to DataFrame

        This is the main entry point for adding exhaustion detection to your data

        Args:
            data: DataFrame with OHLC data
            cusum_lookback: Lookback for CUSUM
            cusum_threshold: Threshold for changepoint detection
            velocity_smooth_window: Smoothing window for velocity

        Returns:
            DataFrame with added exhaustion columns
        """

        detector = ExhaustionDetector(
            cusum_lookback=cusum_lookback,
            cusum_threshold=cusum_threshold,
            velocity_smooth_window=velocity_smooth_window
        )

        exhaustion_data = detector.calculate(data)

        # Add all exhaustion columns to original dataframe
        for col in exhaustion_data.columns:
            data[col] = exhaustion_data[col]

        return data


# Convenience function for easy import
def add_exhaustion_indicators(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to add exhaustion indicators to DataFrame

    Usage:
        from core.indicators.exhaustion_indicators import add_exhaustion_indicators
        data = add_exhaustion_indicators(data)

    Args:
        data: DataFrame with OHLC data
        **kwargs: Optional parameters for ExhaustionDetector

    Returns:
        DataFrame with exhaustion indicators added
    """
    return ExhaustionDetector.calculate_all_exhaustion_indicators(data, **kwargs)
