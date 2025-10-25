"""
Volatility Indicators - ATR, Bollinger Bands
"""

import pandas as pd
import numpy as np


def calculate_atr(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series,
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
    
    Returns:
        pd.Series: ATR values
    """
    # Calculate True Range components
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    
    # True Range = max of the three
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR = SMA of True Range
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_bollinger_bands(close: pd.Series,
                              period: int = 20,
                              num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands
    
    Args:
        close: Close prices
        period: Moving average period
        num_std: Number of standard deviations
    
    Returns:
        pd.DataFrame: DataFrame with bb_upper, bb_middle, bb_lower columns
    """
    # Calculate middle band (SMA)
    middle = close.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_width': upper - lower,
        'bb_percent': (close - lower) / (upper - lower)
    })


def calculate_keltner_channels(high: pd.Series,
                               low: pd.Series,
                               close: pd.Series,
                               ema_period: int = 20,
                               atr_period: int = 10,
                               atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Calculate Keltner Channels
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_period: EMA period for middle line
        atr_period: ATR period
        atr_multiplier: ATR multiplier for bands
    
    Returns:
        pd.DataFrame: DataFrame with kc_upper, kc_middle, kc_lower columns
    """
    # Calculate middle line (EMA)
    middle = close.ewm(span=ema_period, adjust=False).mean()
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, atr_period)
    
    # Calculate upper and lower channels
    upper = middle + (atr * atr_multiplier)
    lower = middle - (atr * atr_multiplier)
    
    return pd.DataFrame({
        'kc_upper': upper,
        'kc_middle': middle,
        'kc_lower': lower
    })


def calculate_donchian_channels(high: pd.Series,
                                low: pd.Series,
                                period: int = 20) -> pd.DataFrame:
    """
    Calculate Donchian Channels
    
    Args:
        high: High prices
        low: Low prices
        period: Channel period
    
    Returns:
        pd.DataFrame: DataFrame with dc_upper, dc_middle, dc_lower columns
    """
    # Upper channel = highest high
    upper = high.rolling(window=period).max()
    
    # Lower channel = lowest low
    lower = low.rolling(window=period).min()
    
    # Middle = average of upper and lower
    middle = (upper + lower) / 2
    
    return pd.DataFrame({
        'dc_upper': upper,
        'dc_middle': middle,
        'dc_lower': lower
    })


def calculate_historical_volatility(close: pd.Series,
                                    period: int = 20,
                                    trading_periods: int = 252) -> pd.Series:
    """
    Calculate Historical Volatility (annualized)
    
    Args:
        close: Close prices
        period: Calculation period
        trading_periods: Trading periods per year (252 for daily, 252*24 for hourly)
    
    Returns:
        pd.Series: Historical volatility (%)
    """
    # Calculate log returns
    log_returns = np.log(close / close.shift(1))
    
    # Calculate rolling standard deviation
    volatility = log_returns.rolling(window=period).std() * np.sqrt(trading_periods) * 100
    
    return volatility


def detect_volatility_squeeze(bb_width: pd.Series,
                              kc_width: pd.Series,
                              threshold: float = 1.0) -> pd.Series:
    """
    Detect volatility squeeze (TTM Squeeze)
    
    Squeeze happens when Bollinger Bands are inside Keltner Channels
    
    Args:
        bb_width: Bollinger Bands width
        kc_width: Keltner Channels width
        threshold: Threshold ratio
    
    Returns:
        pd.Series: True when in squeeze
    """
    squeeze = bb_width < (kc_width * threshold)
    return squeeze
