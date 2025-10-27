# indicators/volume.py
"""
Volume-based Indicators

- VWAP (Volume Weighted Average Price) - Institutional benchmark
- OBV (On-Balance Volume) - Cumulative volume trend
- VolumeAnalyzer - Simple volume spike detection

Author: Claude Code
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
from .base import BaseIndicator


class VWAPIndicator(BaseIndicator):
    """
    VWAP (Volume Weighted Average Price)

    Purpose:
    - Institutional benchmark
    - Support/Resistance
    - Trend filter (price > VWAP = bullish)

    Formula:
        VWAP = Sum(Typical Price * Volume) / Sum(Volume)
        Typical Price = (High + Low + Close) / 3

    Characteristics:
    - NO LAG (real-time calculation)
    - Volume-weighted (reflects institutional trading)
    - Resets daily (or per session)

    Usage:
        vwap = VWAPIndicator()
        vwap_values = vwap.calculate(data)

        # Bullish: price > VWAP
        # Bearish: price < VWAP
    """

    def __init__(self, reset_period: str = None):
        """
        Initialize VWAP

        Args:
            reset_period: Reset period ('D' = daily, None = cumulative)
        """
        super().__init__(name='VWAP', reset_period=reset_period)
        self.reset_period = reset_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.Series: VWAP values
        """
        # Typical Price
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # Price * Volume
        pv = typical_price * data['volume']

        if self.reset_period:
            # Reset VWAP daily (or per period)
            cum_pv = pv.groupby(pd.Grouper(freq=self.reset_period)).cumsum()
            cum_vol = data['volume'].groupby(pd.Grouper(freq=self.reset_period)).cumsum()
        else:
            # Cumulative VWAP (entire period)
            cum_pv = pv.cumsum()
            cum_vol = data['volume'].cumsum()

        vwap = cum_pv / cum_vol

        return vwap

    def get_bands(self, data: pd.DataFrame, std_multiplier: float = 1.0) -> pd.DataFrame:
        """
        Calculate VWAP bands (similar to Bollinger Bands)

        Args:
            data: OHLCV DataFrame
            std_multiplier: Standard deviation multiplier

        Returns:
            pd.DataFrame: Columns [vwap, upper_band, lower_band]
        """
        vwap = self.calculate(data)
        typical_price = (data['high'] + data['low'] + data['close']) / 3

        # Calculate standard deviation
        if self.reset_period:
            std = typical_price.groupby(pd.Grouper(freq=self.reset_period)).transform(
                lambda x: x.expanding().std()
            )
        else:
            std = typical_price.expanding().std()

        upper_band = vwap + (std * std_multiplier)
        lower_band = vwap - (std * std_multiplier)

        return pd.DataFrame({
            'vwap': vwap,
            'upper_band': upper_band,
            'lower_band': lower_band
        }, index=data.index)


class OBVIndicator(BaseIndicator):
    """
    OBV (On-Balance Volume)

    Purpose:
    - Leading volume indicator
    - Accumulation/Distribution detection
    - Divergence signals

    Formula:
        if Close > Close_prev: OBV = OBV_prev + Volume
        if Close < Close_prev: OBV = OBV_prev - Volume
        if Close = Close_prev: OBV = OBV_prev

    Characteristics:
    - Cumulative volume
    - Leading indicator (moves before price)
    - Divergence very powerful

    Usage:
        obv = OBVIndicator()
        obv_values = obv.calculate(data)

        # Bullish: OBV rising
        # Bearish: OBV falling
        # Divergence: Price up, OBV down → weak rally
    """

    def __init__(self):
        """Initialize OBV"""
        super().__init__(name='OBV')

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate OBV

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.Series: OBV values
        """
        close = data['close']
        volume = data['volume']

        # Price direction
        price_diff = close.diff()

        # Volume direction
        volume_direction = pd.Series(0, index=data.index)
        volume_direction[price_diff > 0] = volume[price_diff > 0]
        volume_direction[price_diff < 0] = -volume[price_diff < 0]

        # Cumulative sum
        obv = volume_direction.cumsum()

        return obv

    def get_obv_sma(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate OBV with SMA for trend detection

        Args:
            data: OHLCV DataFrame
            period: SMA period

        Returns:
            pd.DataFrame: Columns [obv, obv_sma]
        """
        obv = self.calculate(data)
        obv_sma = obv.rolling(window=period).mean()

        return pd.DataFrame({
            'obv': obv,
            'obv_sma': obv_sma,
            'obv_above_sma': obv > obv_sma
        }, index=data.index)

    def detect_divergence(self, data: pd.DataFrame, lookback: int = 14) -> pd.Series:
        """
        Detect bullish/bearish divergence

        Args:
            data: OHLCV DataFrame
            lookback: Lookback period for detecting highs/lows

        Returns:
            pd.Series: Divergence signals (1=bullish, -1=bearish, 0=none)
        """
        obv = self.calculate(data)
        close = data['close']

        signals = pd.Series(0, index=data.index)

        for i in range(lookback, len(data)):
            # Get recent highs/lows
            recent_close = close.iloc[i-lookback:i+1]
            recent_obv = obv.iloc[i-lookback:i+1]

            # Bullish divergence: Price lower low, OBV higher low
            if recent_close.iloc[-1] == recent_close.min():  # Price at low
                if recent_obv.iloc[-1] > recent_obv.min():  # OBV not at low
                    signals.iloc[i] = 1  # Bullish divergence

            # Bearish divergence: Price higher high, OBV lower high
            if recent_close.iloc[-1] == recent_close.max():  # Price at high
                if recent_obv.iloc[-1] < recent_obv.max():  # OBV not at high
                    signals.iloc[i] = -1  # Bearish divergence

        return signals


class VolumeAnalyzer(BaseIndicator):
    """
    Simple Volume Analysis

    Purpose:
    - Detect volume spikes (confirmation)
    - Average volume calculation

    Usage:
        vol_analyzer = VolumeAnalyzer(period=20)
        analysis = vol_analyzer.calculate(data)

        # analysis contains: avg_volume, volume_ratio, is_spike
    """

    def __init__(self, period: int = 20, spike_threshold: float = 1.5):
        """
        Initialize Volume Analyzer

        Args:
            period: Period for average volume
            spike_threshold: Threshold for spike (e.g., 1.5 = 150% of avg)
        """
        super().__init__(name='VolumeAnalyzer', period=period, spike_threshold=spike_threshold)
        self.period = period
        self.spike_threshold = spike_threshold

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume analysis

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.DataFrame: Columns [avg_volume, volume_ratio, is_spike]
        """
        volume = data['volume']

        # Average volume
        avg_volume = volume.rolling(window=self.period).mean()

        # Volume ratio
        volume_ratio = volume / avg_volume

        # Spike detection
        is_spike = volume_ratio > self.spike_threshold

        return pd.DataFrame({
            'avg_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'is_spike': is_spike,
            'spike_strength': (volume_ratio - 1) * 100  # % above average
        }, index=data.index)

    def get_volume_score(self, current_volume: float, avg_volume: float) -> int:
        """
        Get volume score for confluence

        Args:
            current_volume: Current candle volume
            avg_volume: Average volume

        Returns:
            int: Score (0-15 points)
        """
        if avg_volume == 0:
            return 0

        ratio = current_volume / avg_volume

        if ratio > 2.0:
            return 15  # Very strong spike
        elif ratio > 1.5:
            return 10  # Strong spike
        elif ratio > 1.2:
            return 5   # Moderate increase
        else:
            return 0   # No spike


# Future volume indicators can be added here:
# class MFIIndicator(BaseIndicator):
#     """Money Flow Index - Volume-based RSI"""
#     pass
#
# class CMFIndicator(BaseIndicator):
#     """Chaikin Money Flow - Buying/selling pressure"""
#     pass
#
# class VolumeProfileIndicator(BaseIndicator):
#     """Volume Profile - Volume at price levels"""
#     pass
