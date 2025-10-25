# indicators/__init__.py
"""
Technical Indicators Module

Extensible architecture for adding/removing indicators easily.

Usage:
    from indicators import ATRIndicator, VWAPIndicator, OBVIndicator

    atr = ATRIndicator(period=14)
    vwap = VWAPIndicator()
    obv = OBVIndicator()

    # Calculate
    atr_values = atr.calculate(data)
    vwap_values = vwap.calculate(data)
    obv_values = obv.calculate(data)

Author: Claude Code
Date: 2025-10-24
"""

from .base import BaseIndicator, IndicatorRegistry
from .volatility import ATRIndicator
from .volume import VWAPIndicator, OBVIndicator, VolumeAnalyzer
from .trend import ADXIndicator
from .confluence import ConfluenceScorer

__all__ = [
    'BaseIndicator',
    'IndicatorRegistry',
    'ATRIndicator',
    'VWAPIndicator',
    'OBVIndicator',
    'VolumeAnalyzer',
    'ADXIndicator',
    'ConfluenceScorer',
]
