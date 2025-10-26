# core/microstructure/hurst_exponent.py
"""
Hurst Exponent Analysis - Đo trend strength và mean reversion tendency

Hurst Exponent (H):
- H < 0.5: Mean reverting (anti-persistent) → Giá đuối, sắp đảo chiều
- H = 0.5: Random walk (no memory)
- H > 0.5: Trending (persistent) → Giá còn mạnh

Critical for detecting:
- Exhaustion: H dropping from >0.5 to <0.5
- Trend continuation: H stable >0.6
- Reversal zones: H <0.4
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats


class HurstExponentAnalyzer:
    """
    Hurst Exponent Analyzer

    Attributes:
        window_size: Rolling window for H calculation (adaptive)
        min_window: Minimum window size
        max_window: Maximum window size
    """

    def __init__(self, window_size: int = 100, min_window: int = 50,
                 max_window: int = 200):
        """
        Initialize Hurst Exponent Analyzer

        Args:
            window_size: Default rolling window
            min_window: Minimum window size
            max_window: Maximum window size
        """
        self.window_size = window_size
        self.min_window = min_window
        self.max_window = max_window

    def calculate_hurst_rs(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst Exponent using R/S (Rescaled Range) method

        Classic method by Mandelbrot & Wallis

        Args:
            prices: Price array

        Returns:
            float: Hurst exponent (0-1)
        """
        if len(prices) < 10:
            return 0.5  # Return neutral for insufficient data

        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])

        if len(log_returns) < 10:
            return 0.5

        # Calculate mean
        mean_return = np.mean(log_returns)

        # Calculate cumulative deviations
        deviations = log_returns - mean_return
        cumulative_deviations = np.cumsum(deviations)

        # Calculate range
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)

        # Calculate standard deviation
        S = np.std(log_returns, ddof=1)

        if S == 0 or R == 0:
            return 0.5

        # R/S ratio
        rs = R / S

        # Hurst exponent: E[R/S] = c * n^H
        # H = log(R/S) / log(n)
        n = len(log_returns)
        H = np.log(rs) / np.log(n)

        # Clip to valid range
        H = np.clip(H, 0.0, 1.0)

        return H

    def calculate_hurst_dfa(self, prices: np.ndarray) -> float:
        """
        Calculate Hurst Exponent using DFA (Detrended Fluctuation Analysis)

        More robust than R/S method

        Args:
            prices: Price array

        Returns:
            float: Hurst exponent (0-1)
        """
        if len(prices) < 20:
            return 0.5

        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])

        if len(log_returns) < 20:
            return 0.5

        # Mean-centered cumulative sum
        Y = np.cumsum(log_returns - np.mean(log_returns))

        # Define window sizes (scales)
        min_window = 10
        max_window = len(Y) // 4
        if max_window < min_window:
            return 0.5

        scales = np.logspace(np.log10(min_window), np.log10(max_window), num=10, dtype=int)
        scales = np.unique(scales)

        fluctuations = []

        for scale in scales:
            # Divide into segments
            n_segments = len(Y) // scale
            if n_segments < 2:
                continue

            F_scale = 0.0

            for segment in range(n_segments):
                start = segment * scale
                end = start + scale
                segment_data = Y[start:end]

                # Fit linear trend
                x = np.arange(len(segment_data))
                coeffs = np.polyfit(x, segment_data, 1)
                trend = np.polyval(coeffs, x)

                # Calculate fluctuation
                F_scale += np.sum((segment_data - trend) ** 2)

            F_scale = np.sqrt(F_scale / (n_segments * scale))
            fluctuations.append(F_scale)

        if len(fluctuations) < 3:
            return 0.5

        # Calculate Hurst: F(scale) ~ scale^H
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluctuations = np.log(fluctuations)

        # Linear regression
        slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)

        H = slope

        # Clip to valid range
        H = np.clip(H, 0.0, 1.0)

        return H

    def calculate_rolling_hurst(self, prices: pd.Series, method: str = 'rs') -> pd.Series:
        """
        Calculate rolling Hurst exponent

        Args:
            prices: Price series
            method: 'rs' or 'dfa'

        Returns:
            pd.Series: Rolling Hurst exponent
        """
        hurst_values = []

        calc_func = self.calculate_hurst_rs if method == 'rs' else self.calculate_hurst_dfa

        for i in range(len(prices)):
            if i < self.min_window:
                hurst_values.append(0.5)
                continue

            # Adaptive window based on volatility
            start_idx = max(0, i - self.window_size)
            window_prices = prices.iloc[start_idx:i+1].values

            H = calc_func(window_prices)
            hurst_values.append(H)

        return pd.Series(hurst_values, index=prices.index)

    def detect_exhaustion_signals(self, hurst_series: pd.Series,
                                  threshold_reverting: float = 0.4,
                                  threshold_trending: float = 0.6) -> pd.Series:
        """
        Detect exhaustion signals from Hurst exponent

        Signals:
        - 1: Bullish exhaustion ending (H crossing above 0.4)
        - -1: Bearish exhaustion ending (H dropping below 0.6)
        - 0: No signal

        Args:
            hurst_series: Rolling Hurst values
            threshold_reverting: H threshold for mean reversion
            threshold_trending: H threshold for trending

        Returns:
            pd.Series: Exhaustion signals
        """
        signals = pd.Series(0, index=hurst_series.index)

        for i in range(1, len(hurst_series)):
            h_prev = hurst_series.iloc[i-1]
            h_curr = hurst_series.iloc[i]

            # Bullish: H crossing from below to above threshold (exiting exhaustion)
            if h_prev < threshold_reverting and h_curr >= threshold_reverting:
                signals.iloc[i] = 1

            # Bearish: H crossing from above to below threshold (entering exhaustion)
            if h_prev > threshold_trending and h_curr <= threshold_trending:
                signals.iloc[i] = -1

        return signals

    def get_market_regime(self, hurst_value: float) -> str:
        """
        Classify market regime based on Hurst exponent

        Args:
            hurst_value: Hurst exponent value

        Returns:
            str: Market regime
        """
        if hurst_value < 0.4:
            return 'STRONG_MEAN_REVERSION'  # Giá đuối, high probability reversal
        elif hurst_value < 0.5:
            return 'MEAN_REVERTING'  # Slight mean reversion tendency
        elif hurst_value < 0.6:
            return 'RANDOM_WALK'  # No clear trend/reversion
        elif hurst_value < 0.7:
            return 'TRENDING'  # Clear trend
        else:
            return 'STRONG_TREND'  # Very strong trend

    def calculate_hurst_momentum(self, hurst_series: pd.Series, window: int = 10) -> pd.Series:
        """
        Calculate momentum of Hurst exponent

        Positive momentum = trend strengthening
        Negative momentum = trend weakening (exhaustion)

        Args:
            hurst_series: Rolling Hurst values
            window: Window for momentum calculation

        Returns:
            pd.Series: Hurst momentum
        """
        # Rate of change
        hurst_roc = hurst_series.diff(window)

        return hurst_roc

    def get_exhaustion_score(self, hurst_value: float, hurst_momentum: float) -> float:
        """
        Calculate exhaustion score (0-1)

        Higher score = higher probability of exhaustion/reversal

        Args:
            hurst_value: Current Hurst exponent
            hurst_momentum: Hurst momentum (ROC)

        Returns:
            float: Exhaustion score (0-1)
        """
        score = 0.0

        # Component 1: Low Hurst value (mean reverting)
        if hurst_value < 0.5:
            score += (0.5 - hurst_value) / 0.5 * 0.5  # Max 0.5 points

        # Component 2: Negative Hurst momentum (weakening trend)
        if hurst_momentum < 0:
            score += min(abs(hurst_momentum) * 5, 0.5)  # Max 0.5 points

        return min(score, 1.0)

    def analyze_price_series(self, prices: pd.Series, method: str = 'rs') -> dict:
        """
        Complete Hurst analysis of price series

        Args:
            prices: Price series
            method: 'rs' or 'dfa'

        Returns:
            dict: Analysis results
        """
        # Calculate rolling Hurst
        hurst_series = self.calculate_rolling_hurst(prices, method=method)

        # Calculate Hurst momentum
        hurst_momentum = self.calculate_hurst_momentum(hurst_series)

        # Detect exhaustion signals
        exhaustion_signals = self.detect_exhaustion_signals(hurst_series)

        # Current values
        current_hurst = hurst_series.iloc[-1]
        current_momentum = hurst_momentum.iloc[-1]
        current_regime = self.get_market_regime(current_hurst)
        exhaustion_score = self.get_exhaustion_score(current_hurst, current_momentum)

        return {
            'hurst_series': hurst_series,
            'hurst_momentum': hurst_momentum,
            'exhaustion_signals': exhaustion_signals,
            'current_hurst': current_hurst,
            'current_momentum': current_momentum,
            'current_regime': current_regime,
            'exhaustion_score': exhaustion_score,
            'is_mean_reverting': current_hurst < 0.5,
            'is_trending': current_hurst > 0.6
        }
