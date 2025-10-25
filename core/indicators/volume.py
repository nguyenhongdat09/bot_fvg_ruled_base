"""
Volume Indicators - Volume analysis
"""

import pandas as pd
import numpy as np


def calculate_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Volume Moving Average
    
    Args:
        volume: Volume series
        period: MA period
    
    Returns:
        pd.Series: Volume MA
    """
    return volume.rolling(window=period).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV)
    
    Args:
        close: Close prices
        volume: Volume
    
    Returns:
        pd.Series: OBV values
    """
    # Calculate price direction
    direction = pd.Series(0, index=close.index)
    direction[close > close.shift(1)] = 1
    direction[close < close.shift(1)] = -1
    
    # Calculate OBV
    obv = (direction * volume).cumsum()
    
    return obv


def calculate_vwap(high: pd.Series,
                   low: pd.Series,
                   close: pd.Series,
                   volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
    
    Returns:
        pd.Series: VWAP values
    """
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate VWAP
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    return vwap


def calculate_mfi(high: pd.Series,
                  low: pd.Series,
                  close: pd.Series,
                  volume: pd.Series,
                  period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        period: MFI period
    
    Returns:
        pd.Series: MFI values (0-100)
    """
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate money flow
    money_flow = typical_price * volume
    
    # Separate positive and negative money flow
    positive_flow = pd.Series(0.0, index=close.index)
    negative_flow = pd.Series(0.0, index=close.index)
    
    positive_flow[typical_price > typical_price.shift(1)] = money_flow
    negative_flow[typical_price < typical_price.shift(1)] = money_flow
    
    # Calculate money flow ratio
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfr = positive_mf / negative_mf
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi


def calculate_volume_profile(close: pd.Series,
                             volume: pd.Series,
                             num_bins: int = 20) -> pd.DataFrame:
    """
    Calculate Volume Profile
    
    Args:
        close: Close prices
        volume: Volume
        num_bins: Number of price bins
    
    Returns:
        pd.DataFrame: Volume profile with price levels and volumes
    """
    # Create price bins
    price_min = close.min()
    price_max = close.max()
    bins = np.linspace(price_min, price_max, num_bins + 1)
    
    # Assign each price to a bin
    price_bins = pd.cut(close, bins=bins, include_lowest=True)
    
    # Sum volume for each bin
    volume_profile = volume.groupby(price_bins).sum()
    
    return pd.DataFrame({
        'price_level': [interval.mid for interval in volume_profile.index],
        'volume': volume_profile.values
    })


def detect_volume_spike(volume: pd.Series,
                       ma_period: int = 20,
                       threshold: float = 2.0) -> pd.Series:
    """
    Detect volume spikes
    
    Args:
        volume: Volume series
        ma_period: MA period for comparison
        threshold: Spike threshold (e.g., 2.0 = 200% of average)
    
    Returns:
        pd.Series: True when volume spike detected
    """
    volume_ma = calculate_volume_ma(volume, ma_period)
    spike = volume > (volume_ma * threshold)
    
    return spike


def calculate_accumulation_distribution(high: pd.Series,
                                       low: pd.Series,
                                       close: pd.Series,
                                       volume: pd.Series) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
    
    Returns:
        pd.Series: A/D Line values
    """
    # Calculate Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)  # Handle division by zero
    
    # Calculate Money Flow Volume
    mfv = mfm * volume
    
    # Calculate A/D Line (cumulative)
    ad_line = mfv.cumsum()
    
    return ad_line
