"""
Trend Indicators - EMA, SMA, Trend detection
"""

import pandas as pd
import numpy as np


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        data: Price series (usually close prices)
        period: EMA period
    
    Returns:
        pd.Series: EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average
    
    Args:
        data: Price series
        period: SMA period
    
    Returns:
        pd.Series: SMA values
    """
    return data.rolling(window=period).mean()


def calculate_multiple_emas(data: pd.Series, periods: list) -> pd.DataFrame:
    """
    Calculate multiple EMAs at once
    
    Args:
        data: Price series
        periods: List of EMA periods
    
    Returns:
        pd.DataFrame: DataFrame with EMA columns
    """
    df = pd.DataFrame(index=data.index)
    
    for period in periods:
        df[f'ema_{period}'] = calculate_ema(data, period)
    
    return df


def detect_trend_ema(close: pd.Series, 
                     fast_period: int = 20,
                     slow_period: int = 50) -> pd.Series:
    """
    Detect trend using EMA crossover
    
    Args:
        close: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
    
    Returns:
        pd.Series: 1 for uptrend, -1 for downtrend, 0 for neutral
    """
    fast_ema = calculate_ema(close, fast_period)
    slow_ema = calculate_ema(close, slow_period)
    
    trend = pd.Series(0, index=close.index)
    trend[fast_ema > slow_ema] = 1   # Uptrend
    trend[fast_ema < slow_ema] = -1  # Downtrend
    
    return trend


def detect_trend_price_action(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Detect trend using higher highs / lower lows
    
    Args:
        close: Close prices
        period: Lookback period
    
    Returns:
        pd.Series: 1 for uptrend, -1 for downtrend, 0 for sideways
    """
    rolling_max = close.rolling(window=period).max()
    rolling_min = close.rolling(window=period).min()
    
    trend = pd.Series(0, index=close.index)
    
    # Uptrend: price making new highs
    trend[close >= rolling_max.shift(1)] = 1
    
    # Downtrend: price making new lows
    trend[close <= rolling_min.shift(1)] = -1
    
    return trend


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
    
    Returns:
        pd.DataFrame: DataFrame with ADX, +DI, -DI columns
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    # Smooth with EMA
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return pd.DataFrame({
        'adx': adx,
        'plus_di': plus_di,
        'minus_di': minus_di
    })
