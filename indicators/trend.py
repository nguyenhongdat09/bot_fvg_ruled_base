# indicators/trend.py
"""
Trend Indicators

- ADX (Average Directional Index) - Trend strength filter

Author: Claude Code
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
from .base import BaseIndicator


class ADXIndicator(BaseIndicator):
    """
    ADX (Average Directional Index)

    Purpose:
    - Measure trend STRENGTH (not direction!)
    - Filter trending vs ranging markets
    - Enable/disable strategies based on market state

    Formula:
        +DM = High - High_prev (if > 0 and > -DM, else 0)
        -DM = Low_prev - Low (if > 0 and > +DM, else 0)
        TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)

        +DI = EMA(+DM, period) / EMA(TR, period) * 100
        -DI = EMA(-DM, period) / EMA(TR, period) * 100

        DX = |(+DI - (-DI))| / (+DI + (-DI)) * 100
        ADX = EMA(DX, period)

    Interpretation:
        ADX > 25: Strong trend (trade WITH trend)
        ADX < 20: Weak trend / Ranging (avoid trend strategies)
        ADX rising: Trend strengthening
        ADX falling: Trend weakening

    Usage:
        adx = ADXIndicator(period=14)
        result = adx.calculate(data)

        # result contains: adx, plus_di, minus_di

        if result['adx'].iloc[-1] > 25:
            # Trending market - enable trend strategies
            pass
    """

    def __init__(self, period: int = 14):
        """
        Initialize ADX

        Args:
            period: ADX period (default 14)
        """
        super().__init__(name='ADX', period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX, +DI, -DI

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: Columns [adx, plus_di, minus_di, trend_strength]
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = pd.Series(0.0, index=data.index)
        minus_dm = pd.Series(0.0, index=data.index)

        # +DM: High - High_prev (if > 0 and > -DM)
        mask_plus = (high_diff > 0) & (high_diff > low_diff)
        plus_dm[mask_plus] = high_diff[mask_plus]

        # -DM: Low_prev - Low (if > 0 and > +DM)
        mask_minus = (low_diff > 0) & (low_diff > high_diff)
        minus_dm[mask_minus] = low_diff[mask_minus]

        # Calculate TR (True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # Smooth +DM, -DM, TR with EMA
        plus_dm_smooth = plus_dm.ewm(span=self.period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=self.period, adjust=False).mean()
        tr_smooth = tr.ewm(span=self.period, adjust=False).mean()

        # Calculate +DI and -DI
        plus_di = (plus_dm_smooth / tr_smooth) * 100
        minus_di = (minus_dm_smooth / tr_smooth) * 100

        # Calculate DX
        di_sum = plus_di + minus_di
        di_diff = abs(plus_di - minus_di)

        # Avoid division by zero
        dx = pd.Series(0.0, index=data.index)
        mask = di_sum != 0
        dx[mask] = (di_diff[mask] / di_sum[mask]) * 100

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        # Trend strength classification
        trend_strength = pd.Series('WEAK', index=data.index)
        trend_strength[adx >= 20] = 'MODERATE'
        trend_strength[adx >= 25] = 'STRONG'
        trend_strength[adx >= 40] = 'VERY_STRONG'

        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'trend_strength': trend_strength
        }, index=data.index)

    def is_trending(self, data: pd.DataFrame, threshold: float = 25.0) -> pd.Series:
        """
        Check if market is trending

        Args:
            data: OHLCV DataFrame
            threshold: ADX threshold for trending (default 25)

        Returns:
            pd.Series: Boolean (True = trending, False = ranging)
        """
        result = self.calculate(data)
        return result['adx'] >= threshold

    def get_trend_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        Get trend direction based on +DI and -DI

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.Series: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        result = self.calculate(data)

        direction = pd.Series('NEUTRAL', index=data.index)
        direction[result['plus_di'] > result['minus_di']] = 'BULLISH'
        direction[result['minus_di'] > result['plus_di']] = 'BEARISH'

        return direction

    def get_filter_score(self, current_adx: float, threshold: float = 25.0) -> bool:
        """
        Get filter decision for confluence scoring

        Args:
            current_adx: Current ADX value
            threshold: ADX threshold

        Returns:
            bool: True if should trade (trending), False if should skip (ranging)
        """
        return current_adx >= threshold


# Future trend indicators can be added here:
# class SupertrendIndicator(BaseIndicator):
#     """Supertrend - ATR-based trend follower"""
#     pass
#
# class EMAIndicator(BaseIndicator):
#     """Exponential Moving Average"""
#     pass
#
# class IchimokuIndicator(BaseIndicator):
#     """Ichimoku Cloud - Complete system"""
#     pass
