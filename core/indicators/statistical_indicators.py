"""
Statistical Indicators for Enhanced FVG Strategy

Advanced statistical measures replacing basic indicators:
1. Hurst Exponent - Market regime detection (trending vs mean-reverting)
2. Skewness - Measures distribution bias (bullish/bearish)
3. Kurtosis - Detects fat tails (extreme moves)
4. OBV Divergence - Enhanced OBV with divergence detection
5. ATR Percentile - Market regime filter

Author: Claude Code
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from scipy import stats


class StatisticalIndicators:
    """Advanced statistical indicators for market analysis"""

    @staticmethod
    def calculate_hurst_exponent(
        data: pd.DataFrame,
        lookback: int = 100,
        min_lags: int = 2,
        max_lags: int = 20
    ) -> pd.Series:
        """
        Calculate Hurst Exponent using Rescaled Range (R/S) Analysis

        Hurst Exponent measures long-term memory and trend persistence:
        - H < 0.5: Mean reverting (anti-persistent) - prices tend to reverse
        - H = 0.5: Random walk (no memory) - Brownian motion
        - H > 0.5: Trending (persistent) - momentum continues

        For FVG Strategy:
        - FVG + H > 0.6 = Strong trend confirmation → HIGH confidence
        - FVG + H < 0.4 = Mean reversion → SKIP (likely fake breakout)
        - FVG + 0.4 ≤ H ≤ 0.6 = Neutral

        Args:
            data: DataFrame with 'close' prices
            lookback: Rolling window size for calculation
            min_lags: Minimum lag for R/S analysis
            max_lags: Maximum lag for R/S analysis

        Returns:
            Series of Hurst Exponent values (0-1)

        Reference:
            Hurst, H. E. (1951). "Long-term storage capacity of reservoirs"
            Peters, E. E. (1994). "Fractal Market Analysis"
        """
        close = data['close'].values
        hurst_values = []

        for i in range(len(data)):
            if i < lookback:
                hurst_values.append(np.nan)
                continue

            # Get window of returns
            window = close[i-lookback:i]
            returns = np.diff(np.log(window))

            if len(returns) < max_lags:
                hurst_values.append(np.nan)
                continue

            # Calculate R/S statistics for different lags
            lags = range(min_lags, min(max_lags, len(returns) // 2))
            rs_values = []

            for lag in lags:
                # Split returns into chunks of size 'lag'
                n_chunks = len(returns) // lag

                if n_chunks == 0:
                    continue

                rs_chunk = []
                for j in range(n_chunks):
                    chunk = returns[j*lag:(j+1)*lag]

                    if len(chunk) < 2:
                        continue

                    # Mean of chunk
                    mean_chunk = np.mean(chunk)

                    # Mean-adjusted series
                    adjusted = chunk - mean_chunk

                    # Cumulative sum of adjusted series
                    cumsum = np.cumsum(adjusted)

                    # Range
                    R = np.max(cumsum) - np.min(cumsum)

                    # Standard deviation
                    S = np.std(chunk, ddof=1)

                    if S > 0:
                        rs_chunk.append(R / S)

                if len(rs_chunk) > 0:
                    rs_values.append(np.mean(rs_chunk))

            if len(rs_values) < 2:
                hurst_values.append(0.5)  # Default to random walk
                continue

            # Hurst exponent is slope of log(R/S) vs log(lag)
            try:
                log_lags = np.log([lags[i] for i in range(len(rs_values))])
                log_rs = np.log(rs_values)

                # Linear regression
                slope, _ = np.polyfit(log_lags, log_rs, 1)

                # Bound to [0, 1]
                hurst = max(0.0, min(1.0, slope))

                hurst_values.append(hurst)
            except:
                hurst_values.append(0.5)  # Default to random walk on error

        return pd.Series(hurst_values, index=data.index, name='hurst')

    @staticmethod
    def calculate_hurst_regime(hurst: pd.Series) -> pd.Series:
        """
        Classify market regime based on Hurst Exponent

        Args:
            hurst: Series of Hurst Exponent values

        Returns:
            Series with regime classification
            1 = Trending (H > 0.55)
            0 = Random (0.45 ≤ H ≤ 0.55)
            -1 = Mean Reverting (H < 0.45)
        """
        regime = pd.Series(0, index=hurst.index)
        regime[hurst > 0.55] = 1   # Trending
        regime[hurst < 0.45] = -1  # Mean reverting
        return regime

    @staticmethod
    def calculate_linear_regression_deviation(
        data: pd.DataFrame,
        lookback: int = 20,
        return_r2: bool = False
    ) -> pd.Series:
        """
        Calculate Linear Regression Deviation (standardized)

        Measures how far price is from the regression line (statistical fair value).
        Similar to Bollinger Bands but using linear regression instead of SMA.

        For FVG Strategy:
        - Large deviation + FVG = Price far from fair value → Mean reversion setup
        - Small deviation = Price near fair value → Continuation setup

        Args:
            data: DataFrame with 'close' prices
            lookback: Rolling window for regression
            return_r2: If True, also return R-squared values

        Returns:
            Series of standardized deviations (in standard deviations)
            If return_r2=True, returns tuple (deviations, r2_values)

        Example:
            LR_dev = 2.5 → Price is 2.5 std above regression line (overbought)
            LR_dev = -2.0 → Price is 2.0 std below regression line (oversold)
        """
        close = data['close'].values
        deviations = []
        r2_values = [] if return_r2 else None

        for i in range(len(data)):
            if i < lookback:
                deviations.append(np.nan)
                if return_r2:
                    r2_values.append(np.nan)
                continue

            # Get window
            window = close[i-lookback:i+1]  # Include current candle
            x = np.arange(len(window))
            y = window

            # Linear regression
            try:
                # Fit line: y = slope * x + intercept
                coeffs = np.polyfit(x, y, 1)
                slope, intercept = coeffs

                # Predicted values
                y_pred = slope * x + intercept

                # Current price deviation from regression line
                current_price = y[-1]
                predicted_price = y_pred[-1]
                deviation_abs = current_price - predicted_price

                # Standardize by std of residuals
                residuals = y - y_pred
                std_residuals = np.std(residuals)

                if std_residuals > 0:
                    deviation_std = deviation_abs / std_residuals
                else:
                    deviation_std = 0.0

                deviations.append(deviation_std)

                # Calculate R-squared if requested
                if return_r2:
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    r2_values.append(max(0, min(1, r2)))  # Bound to [0, 1]

            except:
                deviations.append(0.0)
                if return_r2:
                    r2_values.append(0.0)

        result = pd.Series(deviations, index=data.index, name='lr_deviation')

        if return_r2:
            r2_series = pd.Series(r2_values, index=data.index, name='r2')
            return result, r2_series
        else:
            return result

    @staticmethod
    def calculate_returns_skewness(
        data: pd.DataFrame,
        lookback: int = 20
    ) -> pd.Series:
        """
        Calculate rolling skewness of returns

        Skewness measures asymmetry in distribution:
        - Positive skew: More small losses + few large gains → Bullish bias
        - Negative skew: More small gains + few large losses → Bearish bias
        - Near zero: Symmetric distribution → No bias

        Args:
            data: DataFrame with 'close'
            lookback: Rolling window size

        Returns:
            Series of skewness values
        """
        # Calculate returns
        returns = data['close'].pct_change()

        # Rolling skewness
        skewness = returns.rolling(window=lookback, min_periods=lookback).apply(
            lambda x: stats.skew(x), raw=True
        )

        return skewness

    @staticmethod
    def calculate_returns_kurtosis(
        data: pd.DataFrame,
        lookback: int = 20
    ) -> pd.Series:
        """
        Calculate rolling kurtosis (excess) of returns

        Kurtosis measures "fat tails" - probability of extreme moves:
        - High kurtosis (>3): Fat tails → High probability of large moves
        - Low kurtosis (<3): Thin tails → Normal distribution
        - Excess kurtosis = kurtosis - 3 (normal distribution = 0)

        Use: High kurtosis + FVG = Strong breakout signal

        Args:
            data: DataFrame with 'close'
            lookback: Rolling window size

        Returns:
            Series of excess kurtosis values
        """
        # Calculate returns
        returns = data['close'].pct_change()

        # Rolling kurtosis (excess)
        kurtosis = returns.rolling(window=lookback, min_periods=lookback).apply(
            lambda x: stats.kurtosis(x, fisher=True), raw=True
        )

        return kurtosis

    @staticmethod
    def calculate_obv_divergence(
        data: pd.DataFrame,
        lookback: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate OBV with divergence detection

        Divergence types:
        - Bullish divergence: Price makes lower low, OBV makes higher low
        - Bearish divergence: Price makes higher high, OBV makes lower high

        Args:
            data: DataFrame with 'close', 'volume'
            lookback: Window for finding swing points

        Returns:
            Tuple of (OBV, divergence_signal)
            divergence_signal: 1 = bullish, -1 = bearish, 0 = none
        """
        # Calculate OBV
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()

        # Find swing highs and lows in price
        price_highs = data['close'].rolling(window=lookback, center=True).max()
        price_lows = data['close'].rolling(window=lookback, center=True).min()

        is_price_high = data['close'] == price_highs
        is_price_low = data['close'] == price_lows

        # Find swing highs and lows in OBV
        obv_highs = obv.rolling(window=lookback, center=True).max()
        obv_lows = obv.rolling(window=lookback, center=True).min()

        is_obv_high = obv == obv_highs
        is_obv_low = obv == obv_lows

        # Detect divergence
        divergence = pd.Series(0, index=data.index)

        for i in range(lookback, len(data) - lookback):
            # Bullish divergence: Price lower low + OBV higher low
            if is_price_low.iloc[i]:
                # Find previous low
                prev_lows = is_price_low.iloc[i-lookback:i]
                if prev_lows.any():
                    prev_idx = prev_lows[::-1].idxmax()

                    if (data['close'].iloc[i] < data['close'].loc[prev_idx] and
                        obv.iloc[i] > obv.loc[prev_idx]):
                        divergence.iloc[i] = 1  # Bullish

            # Bearish divergence: Price higher high + OBV lower high
            if is_price_high.iloc[i]:
                # Find previous high
                prev_highs = is_price_high.iloc[i-lookback:i]
                if prev_highs.any():
                    prev_idx = prev_highs[::-1].idxmax()

                    if (data['close'].iloc[i] > data['close'].loc[prev_idx] and
                        obv.iloc[i] < obv.loc[prev_idx]):
                        divergence.iloc[i] = -1  # Bearish

        return obv, divergence

    @staticmethod
    def calculate_atr_percentile(
        data: pd.DataFrame,
        atr_period: int = 14,
        percentile_lookback: int = 100
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate ATR percentile for market regime detection

        Percentile interpretation:
        - 0-30: Low volatility → Avoid trading (fake signals, ranging market)
        - 30-70: Normal volatility → TRADE (best conditions)
        - 70-100: High volatility → Avoid trading (chaos, whipsaws)

        Args:
            data: DataFrame with 'high', 'low', 'close'
            atr_period: ATR calculation period
            percentile_lookback: Window for percentile calculation

        Returns:
            Tuple of (ATR, ATR_percentile)
        """
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()

        # Calculate percentile rank
        atr_percentile = atr.rolling(window=percentile_lookback, min_periods=percentile_lookback).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100, raw=False
        )

        return atr, atr_percentile

    @staticmethod
    def calculate_all_statistical_indicators(
        data: pd.DataFrame,
        hurst_lookback: int = 100,
        lr_lookback: int = 50,
        skew_lookback: int = 20,
        kurt_lookback: int = 20,
        obv_lookback: int = 14,
        atr_period: int = 14,
        atr_percentile_lookback: int = 100
    ) -> pd.DataFrame:
        """
        Calculate all statistical indicators at once

        Args:
            data: DataFrame with OHLCV data
            hurst_lookback: Window for Hurst Exponent calculation
            lr_lookback: Window for Linear Regression deviation
            skew_lookback: Window for skewness
            kurt_lookback: Window for kurtosis
            obv_lookback: Window for OBV divergence
            atr_period: ATR period
            atr_percentile_lookback: Window for ATR percentile

        Returns:
            DataFrame with all indicators added
        """
        result = data.copy()

        # Hurst Exponent (Market Regime Detection)
        result['hurst'] = StatisticalIndicators.calculate_hurst_exponent(
            data, lookback=hurst_lookback
        )
        result['hurst_regime'] = StatisticalIndicators.calculate_hurst_regime(
            result['hurst']
        )

        # Linear Regression Deviation
        result['lr_deviation'], result['r2'] = StatisticalIndicators.calculate_linear_regression_deviation(
            data, lookback=lr_lookback, return_r2=True
        )

        # Skewness
        result['skewness'] = StatisticalIndicators.calculate_returns_skewness(
            data, lookback=skew_lookback
        )

        # Kurtosis
        result['kurtosis'] = StatisticalIndicators.calculate_returns_kurtosis(
            data, lookback=kurt_lookback
        )

        # OBV Divergence
        result['OBV'], result['OBV_divergence'] = StatisticalIndicators.calculate_obv_divergence(
            data, lookback=obv_lookback
        )

        # ATR Percentile (Market Regime)
        result['ATR'], result['ATR_percentile'] = StatisticalIndicators.calculate_atr_percentile(
            data, atr_period=atr_period, percentile_lookback=atr_percentile_lookback
        )

        return result


# Scoring functions for confluence
class StatisticalScoring:
    """Scoring logic for statistical indicators in confluence system"""

    @staticmethod
    def score_hurst(hurst: float, fvg_direction: str) -> float:
        """
        Score Hurst Exponent alignment with FVG

        Logic:
        FVG = price imbalance from TRENDING move
        Hurst > 0.5 = trending market → FVG more reliable
        Hurst < 0.5 = mean reverting → FVG likely false signal

        Scoring:
        - H > 0.6 + FVG = 100 (strong trending, perfect for FVG)
        - 0.55 < H ≤ 0.6 = 70 (moderate trending)
        - 0.45 ≤ H ≤ 0.55 = 40 (random walk, neutral)
        - 0.4 < H < 0.45 = 20 (weak mean reversion)
        - H ≤ 0.4 = 0 (strong mean reversion, SKIP FVG!)

        Args:
            hurst: Hurst Exponent value (0-1)
            fvg_direction: 'BULLISH_BIAS' or 'BEARISH_BIAS'

        Returns:
            Score 0-100
        """
        if pd.isna(hurst):
            return 50  # Neutral if no data

        # Main scoring based on Hurst value
        if hurst > 0.6:
            # Strong trending - PERFECT for FVG
            score = 100
        elif hurst > 0.55:
            # Moderate trending - good for FVG
            score = 70
        elif hurst >= 0.45:
            # Random walk - neutral
            score = 40
        elif hurst > 0.4:
            # Weak mean reversion - risky
            score = 20
        else:  # hurst <= 0.4
            # Strong mean reversion - SKIP FVG!
            score = 0

        return score

    @staticmethod
    def score_lr_deviation(lr_dev: float, fvg_direction: str, r2: float = None) -> float:
        """
        Score Linear Regression Deviation with FVG

        Logic:
        - Large deviation + FVG in opposite direction = mean reversion setup (HIGH score)
        - Small deviation = price near fair value = continuation (MEDIUM score)

        Examples:
        - LR_dev = +2.5σ (price far above line) + FVG Bearish = 100 (perfect short)
        - LR_dev = -2.0σ (price far below line) + FVG Bullish = 100 (perfect long)
        - LR_dev = 0.5σ (near line) + FVG = 40 (weak setup)

        Args:
            lr_dev: Linear regression deviation in standard deviations
            fvg_direction: 'BULLISH_BIAS' or 'BEARISH_BIAS'
            r2: Optional R-squared value (trend confidence, 0-1)

        Returns:
            Score 0-100
        """
        if pd.isna(lr_dev):
            return 50  # Neutral if no data

        abs_dev = abs(lr_dev)

        # Base score from deviation magnitude
        if abs_dev > 2.5:
            magnitude_score = 100  # Extreme deviation
        elif abs_dev > 2.0:
            magnitude_score = 85
        elif abs_dev > 1.5:
            magnitude_score = 70
        elif abs_dev > 1.0:
            magnitude_score = 55
        elif abs_dev > 0.5:
            magnitude_score = 40
        else:
            magnitude_score = 20  # Near fair value

        # Direction alignment bonus (mean reversion logic)
        # FVG Bearish + price above regression = perfect short
        # FVG Bullish + price below regression = perfect long
        if (fvg_direction == 'BEARISH_BIAS' and lr_dev > 0) or \
           (fvg_direction == 'BULLISH_BIAS' and lr_dev < 0):
            # Mean reversion setup - perfect alignment
            alignment_bonus = 1.0
        elif (fvg_direction == 'BEARISH_BIAS' and lr_dev < 0) or \
             (fvg_direction == 'BULLISH_BIAS' and lr_dev > 0):
            # Continuation setup - weaker signal
            alignment_bonus = 0.6
        else:
            # Neutral
            alignment_bonus = 0.7

        score = magnitude_score * alignment_bonus

        # R-squared bonus (if provided)
        # High R² = strong trend = more reliable regression line
        if r2 is not None and not pd.isna(r2):
            if r2 > 0.8:
                r2_multiplier = 1.1  # +10% bonus
            elif r2 > 0.6:
                r2_multiplier = 1.0  # No change
            elif r2 > 0.4:
                r2_multiplier = 0.9  # -10% penalty
            else:
                r2_multiplier = 0.8  # -20% penalty (weak trend)

            score *= r2_multiplier

        return min(100, max(0, score))

    @staticmethod
    def score_fvg_size_atr(fvg_size: float, atr_value: float) -> float:
        """
        Score FVG size normalized by ATR

        Logic:
        - FVG too small vs ATR = noise (low score)
        - FVG optimal size (0.5-1.5 ATR) = strong imbalance (high score)
        - FVG too large (>2 ATR) = might not fill (medium score)

        Args:
            fvg_size: FVG gap size in pips
            atr_value: Current ATR value in pips

        Returns:
            Score 0-100
        """
        if pd.isna(fvg_size) or pd.isna(atr_value) or atr_value == 0:
            return 50  # Neutral if no data

        # Normalize FVG size by ATR
        fvg_strength = fvg_size / atr_value

        # Scoring based on optimal range
        if 0.5 <= fvg_strength <= 1.5:
            # Optimal range - perfect imbalance strength
            # Peak at 1.0 ATR
            distance_from_1 = abs(fvg_strength - 1.0)
            score = 100 - (distance_from_1 * 20)  # Max 100 at 1.0, 80 at boundaries
        elif 0.3 <= fvg_strength < 0.5:
            # Small but acceptable
            score = 60
        elif 1.5 < fvg_strength <= 2.0:
            # Large but might still fill
            score = 70
        elif fvg_strength > 2.0:
            # Very large - might not fill completely
            score = 50
        elif fvg_strength >= 0.2:
            # Very small - likely noise
            score = 30
        else:
            # Tiny gap - definitely noise
            score = 10

        return score

    @staticmethod
    def score_skewness(skewness: float, fvg_direction: str) -> float:
        """
        Score skewness alignment with FVG

        Logic:
        - Positive skew + Bullish FVG = high score
        - Negative skew + Bearish FVG = high score

        Args:
            skewness: Skewness value
            fvg_direction: 'BULLISH_BIAS' or 'BEARISH_BIAS'

        Returns:
            Score 0-100
        """
        if pd.isna(skewness):
            return 0

        # Magnitude score (stronger bias = higher score)
        abs_skew = abs(skewness)
        if abs_skew > 1.0:
            magnitude_score = 100
        elif abs_skew > 0.5:
            magnitude_score = 70
        elif abs_skew > 0.2:
            magnitude_score = 40
        else:
            magnitude_score = 20

        # Direction alignment
        if (skewness > 0 and fvg_direction == 'BULLISH_BIAS') or \
           (skewness < 0 and fvg_direction == 'BEARISH_BIAS'):
            return magnitude_score
        else:
            # Counter-trend (lower score)
            return magnitude_score * 0.3

    @staticmethod
    def score_kurtosis(kurtosis: float) -> float:
        """
        Score kurtosis (fat tails)

        Logic:
        - High kurtosis = high probability of extreme move = good for breakouts
        - Low kurtosis = normal distribution = lower conviction

        Args:
            kurtosis: Excess kurtosis value

        Returns:
            Score 0-100
        """
        if pd.isna(kurtosis):
            return 0

        # Higher kurtosis = higher score (fat tails = strong moves)
        if kurtosis > 3:
            return 100
        elif kurtosis > 1:
            return 70
        elif kurtosis > 0:
            return 40
        else:
            return 20

    @staticmethod
    def score_obv_divergence(divergence: int, fvg_direction: str) -> float:
        """
        Score OBV divergence

        Logic:
        - Bullish divergence + Bullish FVG = very high score
        - Bearish divergence + Bearish FVG = very high score
        - No divergence = moderate score based on OBV trend

        Args:
            divergence: 1 = bullish, -1 = bearish, 0 = none
            fvg_direction: 'BULLISH_BIAS' or 'BEARISH_BIAS'

        Returns:
            Score 0-100
        """
        if divergence == 0:
            return 50  # Neutral

        # Strong alignment
        if (divergence == 1 and fvg_direction == 'BULLISH_BIAS') or \
           (divergence == -1 and fvg_direction == 'BEARISH_BIAS'):
            return 100  # Perfect divergence signal
        else:
            return 20  # Counter-trend divergence

    @staticmethod
    def score_market_regime(atr_percentile: float) -> float:
        """
        Score market regime based on ATR percentile

        Logic:
        - 30-70 percentile = BEST (normal volatility, clear trends)
        - <30 or >70 = AVOID (too quiet or too chaotic)

        Args:
            atr_percentile: ATR percentile (0-100)

        Returns:
            Score 0-100
        """
        if pd.isna(atr_percentile):
            return 50

        # Optimal zone: 30-70 percentile
        if 30 <= atr_percentile <= 70:
            # Peak score at 50th percentile
            distance_from_50 = abs(atr_percentile - 50)
            return 100 - distance_from_50
        elif atr_percentile < 30:
            # Too low volatility (ranging, fake signals)
            return atr_percentile / 30 * 30  # Max 30 score
        else:  # > 70
            # Too high volatility (chaos, whipsaws)
            return (100 - atr_percentile) / 30 * 30  # Max 30 score
