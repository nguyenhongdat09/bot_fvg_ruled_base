# indicators/volatility.py
"""
Volatility Indicators

ATR (Average True Range) - Essential for risk management

Author: Claude Code
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
from .base import BaseIndicator


class ATRIndicator(BaseIndicator):
    """
    ATR (Average True Range) Indicator

    Purpose:
    - Measure volatility
    - Position sizing
    - Dynamic SL/TP

    Formula:
        TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = EMA(TR, period)

    Usage:
        atr = ATRIndicator(period=14)
        atr_values = atr.calculate(data)

        # Position sizing
        lot_size = (account * risk_pct) / (atr_values[-1] * multiplier)

        # Dynamic SL/TP
        sl = entry - (atr_values[-1] * 1.5)
        tp = entry + (atr_values[-1] * 3)
    """

    def __init__(self, period: int = 14):
        """
        Initialize ATR

        Args:
            period: ATR period (default 14)
        """
        super().__init__(name='ATR', period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR

        Args:
            data: OHLCV DataFrame

        Returns:
            pd.Series: ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # ATR = EMA of TR
        atr = tr.ewm(span=self.period, adjust=False).mean()

        return atr

    def get_position_size(self, account_size: float, risk_percent: float,
                         current_atr: float, multiplier: float = 2.0) -> float:
        """
        Calculate position size based on ATR

        Args:
            account_size: Account balance
            risk_percent: Risk percentage (e.g., 0.01 = 1%)
            current_atr: Current ATR value
            multiplier: ATR multiplier for SL distance

        Returns:
            float: Position size (lots)

        Example:
            atr_values = atr.calculate(data)
            current_atr = atr_values.iloc[-1]
            lot_size = atr.get_position_size(10000, 0.01, current_atr)
        """
        risk_amount = account_size * risk_percent
        sl_distance = current_atr * multiplier

        # Position size = risk / sl_distance
        position_size = risk_amount / sl_distance

        return position_size

    def get_sl_tp_levels(self, entry_price: float, current_atr: float,
                         direction: str = 'BUY',
                         sl_multiplier: float = 1.5,
                         tp_multiplier: float = 3.0) -> dict:
        """
        Calculate dynamic SL/TP based on ATR

        Args:
            entry_price: Entry price
            current_atr: Current ATR value
            direction: 'BUY' or 'SELL'
            sl_multiplier: ATR multiplier for stop loss
            tp_multiplier: ATR multiplier for take profit

        Returns:
            dict: {'sl': float, 'tp': float, 'rr_ratio': float}

        Example:
            levels = atr.get_sl_tp_levels(1.10000, 0.00050, 'BUY')
            # {'sl': 1.09925, 'tp': 1.10150, 'rr_ratio': 2.0}
        """
        sl_distance = current_atr * sl_multiplier
        tp_distance = current_atr * tp_multiplier

        if direction == 'BUY':
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # SELL
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        rr_ratio = tp_multiplier / sl_multiplier

        return {
            'sl': round(sl, 5),
            'tp': round(tp, 5),
            'rr_ratio': round(rr_ratio, 2),
            'sl_distance': round(sl_distance, 5),
            'tp_distance': round(tp_distance, 5)
        }


# Future volatility indicators can be added here:
# class BollingerBandsIndicator(BaseIndicator):
#     """Bollinger Bands - volatility envelope"""
#     pass
#
# class KeltnerChannelIndicator(BaseIndicator):
#     """Keltner Channel - ATR-based bands"""
#     pass
