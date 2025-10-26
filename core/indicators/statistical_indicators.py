"""
Statistical Indicators for Enhanced FVG Strategy

Replaces basic indicators (VWAP, Volume Spike) with advanced statistical measures:
1. Volume Profile + POC (Point of Control) - Finds highest volume price level
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
    def calculate_volume_profile_poc(
        data: pd.DataFrame,
        lookback: int = 50,
        price_bins: int = 20
    ) -> pd.Series:
        """
        Calculate Point of Control (POC) from Volume Profile

        POC = Price level with highest volume concentration
        Better than VWAP because it shows where most trading happened,
        not average price

        Args:
            data: DataFrame with 'close', 'high', 'low', 'volume'
            lookback: Number of candles to look back
            price_bins: Number of price bins for profile

        Returns:
            Series of POC prices
        """
        poc_values = []

        for i in range(len(data)):
            if i < lookback:
                poc_values.append(np.nan)
                continue

            # Get lookback window
            window = data.iloc[i-lookback:i]

            # Find price range
            price_min = window['low'].min()
            price_max = window['high'].max()

            # Create price bins
            bins = np.linspace(price_min, price_max, price_bins + 1)

            # Allocate volume to bins based on candle range
            volume_profile = np.zeros(price_bins)

            for _, candle in window.iterrows():
                # Find which bins this candle covers
                candle_low = candle['low']
                candle_high = candle['high']
                candle_volume = candle['volume']

                # Distribute volume to bins proportionally
                for j in range(price_bins):
                    bin_low = bins[j]
                    bin_high = bins[j + 1]

                    # Calculate overlap between candle and bin
                    overlap_low = max(bin_low, candle_low)
                    overlap_high = min(bin_high, candle_high)

                    if overlap_high > overlap_low:
                        # Proportional volume allocation
                        overlap_ratio = (overlap_high - overlap_low) / (candle_high - candle_low + 1e-10)
                        volume_profile[j] += candle_volume * overlap_ratio

            # POC = midpoint of bin with highest volume
            poc_bin_idx = np.argmax(volume_profile)
            poc_price = (bins[poc_bin_idx] + bins[poc_bin_idx + 1]) / 2

            poc_values.append(poc_price)

        return pd.Series(poc_values, index=data.index, name='POC')

    @staticmethod
    def calculate_poc_distance(
        data: pd.DataFrame,
        poc: pd.Series
    ) -> pd.Series:
        """
        Calculate distance from current price to POC

        Returns:
            Distance in pips (positive = above POC, negative = below POC)
        """
        pip_value = 0.0001
        distance = (data['close'] - poc) / pip_value
        return distance

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
        poc_lookback: int = 50,
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

        Returns:
            DataFrame with all indicators added
        """
        result = data.copy()

        # Volume Profile + POC
        result['POC'] = StatisticalIndicators.calculate_volume_profile_poc(
            data, lookback=poc_lookback
        )
        result['POC_distance'] = StatisticalIndicators.calculate_poc_distance(
            data, result['POC']
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
    def score_poc_alignment(poc_distance: float, fvg_direction: str) -> float:
        """
        Score POC alignment with FVG

        Logic:
        - Bullish FVG near POC support = high score
        - Bearish FVG near POC resistance = high score
        - Far from POC = lower score

        Args:
            poc_distance: Distance to POC in pips (+ above, - below)
            fvg_direction: 'BULLISH_BIAS' or 'BEARISH_BIAS'

        Returns:
            Score 0-100
        """
        abs_distance = abs(poc_distance)

        # Close to POC = higher score
        if abs_distance < 10:  # Within 10 pips
            proximity_score = 100
        elif abs_distance < 30:  # 10-30 pips
            proximity_score = 70
        elif abs_distance < 50:  # 30-50 pips
            proximity_score = 40
        else:  # > 50 pips
            proximity_score = 20

        # Direction alignment bonus
        if fvg_direction == 'BULLISH_BIAS' and poc_distance < 0:
            # Price below POC, expecting bounce up
            direction_bonus = 20
        elif fvg_direction == 'BEARISH_BIAS' and poc_distance > 0:
            # Price above POC, expecting rejection down
            direction_bonus = 20
        else:
            direction_bonus = 0

        return min(100, proximity_score + direction_bonus)

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
