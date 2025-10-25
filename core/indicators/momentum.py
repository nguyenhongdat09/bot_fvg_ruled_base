"""
Momentum Indicators - RSI, Stochastic, MACD
"""

import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        close: Close prices
        period: RSI period (default 14)
    
    Returns:
        pd.Series: RSI values (0-100)
    """
    # Calculate price changes
    delta = close.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_stochastic(high: pd.Series, 
                         low: pd.Series, 
                         close: pd.Series,
                         k_period: int = 14,
                         d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period (smoothing)
    
    Returns:
        pd.DataFrame: DataFrame with %K and %D columns
    """
    # Calculate %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    # Calculate %D (smoothed %K)
    d = k.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'stoch_k': k,
        'stoch_d': d
    })


def calculate_macd(close: pd.Series,
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        close: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
    
    Returns:
        pd.DataFrame: DataFrame with MACD, signal, and histogram columns
    """
    # Calculate MACD line
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    })


def calculate_roc(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Rate of Change (ROC)
    
    Args:
        close: Close prices
        period: ROC period
    
    Returns:
        pd.Series: ROC values (percentage)
    """
    roc = ((close - close.shift(period)) / close.shift(period)) * 100
    return roc


def calculate_cci(high: pd.Series, 
                  low: pd.Series, 
                  close: pd.Series,
                  period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: CCI period
    
    Returns:
        pd.Series: CCI values
    """
    # Calculate typical price
    tp = (high + low + close) / 3
    
    # Calculate SMA of typical price
    sma_tp = tp.rolling(window=period).mean()
    
    # Calculate mean deviation
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    
    # Calculate CCI
    cci = (tp - sma_tp) / (0.015 * mad)
    
    return cci


def detect_divergence(price: pd.Series, 
                      indicator: pd.Series,
                      lookback: int = 14) -> pd.DataFrame:
    """
    Detect bullish/bearish divergences
    
    Args:
        price: Price series
        indicator: Indicator series (RSI, MACD, etc.)
        lookback: Lookback period for peaks/troughs
    
    Returns:
        pd.DataFrame: DataFrame with bullish_div and bearish_div columns
    """
    # Find local peaks and troughs
    price_peaks = price.rolling(window=lookback, center=True).max() == price
    price_troughs = price.rolling(window=lookback, center=True).min() == price
    
    indicator_peaks = indicator.rolling(window=lookback, center=True).max() == indicator
    indicator_troughs = indicator.rolling(window=lookback, center=True).min() == indicator
    
    # Detect divergences
    bullish_div = pd.Series(False, index=price.index)
    bearish_div = pd.Series(False, index=price.index)
    
    # Bullish divergence: price making lower lows, indicator making higher lows
    for i in range(lookback, len(price)):
        if price_troughs.iloc[i]:
            prev_trough_idx = price_troughs.iloc[i-lookback:i].idxmax()
            if prev_trough_idx and price.iloc[i] < price[prev_trough_idx]:
                if indicator.iloc[i] > indicator[prev_trough_idx]:
                    bullish_div.iloc[i] = True
    
    # Bearish divergence: price making higher highs, indicator making lower highs
    for i in range(lookback, len(price)):
        if price_peaks.iloc[i]:
            prev_peak_idx = price_peaks.iloc[i-lookback:i].idxmax()
            if prev_peak_idx and price.iloc[i] > price[prev_peak_idx]:
                if indicator.iloc[i] < indicator[prev_peak_idx]:
                    bearish_div.iloc[i] = True
    
    return pd.DataFrame({
        'bullish_divergence': bullish_div,
        'bearish_divergence': bearish_div
    })
