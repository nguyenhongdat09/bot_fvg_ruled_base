# indicators/base.py
"""
Base Indicator Class & Registry

Provides extensible architecture for adding/removing indicators.

Design:
- BaseIndicator: Abstract base class for all indicators
- IndicatorRegistry: Manage and discover indicators

Author: Claude Code
Date: 2025-10-24
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseIndicator(ABC):
    """
    Base class for all technical indicators

    Architecture allows easy extension:
    1. Inherit from BaseIndicator
    2. Implement calculate() method
    3. Auto-registered via IndicatorRegistry

    Example:
        class MyIndicator(BaseIndicator):
            def __init__(self, period=14):
                super().__init__(name='MyIndicator', period=period)

            def calculate(self, data: pd.DataFrame) -> pd.Series:
                return data['close'].rolling(self.period).mean()
    """

    def __init__(self, name: str, **params):
        """
        Initialize indicator

        Args:
            name: Indicator name
            **params: Indicator parameters (e.g., period=14)
        """
        self.name = name
        self.params = params

        # Register this indicator
        IndicatorRegistry.register(self)

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate indicator values

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume

        Returns:
            pd.Series: Indicator values (same index as input data)
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """Get indicator parameters"""
        return self.params.copy()

    def get_name(self) -> str:
        """Get indicator name"""
        return self.name

    def __repr__(self) -> str:
        """String representation"""
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"


class IndicatorRegistry:
    """
    Registry for managing indicators

    Allows dynamic discovery and configuration of indicators.

    Usage:
        # Get all registered indicators
        indicators = IndicatorRegistry.get_all()

        # Get specific indicator
        atr = IndicatorRegistry.get('ATR')

        # List all indicator names
        names = IndicatorRegistry.list_names()
    """

    _indicators: Dict[str, BaseIndicator] = {}

    @classmethod
    def register(cls, indicator: BaseIndicator):
        """
        Register an indicator

        Args:
            indicator: BaseIndicator instance
        """
        cls._indicators[indicator.name] = indicator

    @classmethod
    def get(cls, name: str) -> Optional[BaseIndicator]:
        """
        Get indicator by name

        Args:
            name: Indicator name

        Returns:
            BaseIndicator or None if not found
        """
        return cls._indicators.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, BaseIndicator]:
        """Get all registered indicators"""
        return cls._indicators.copy()

    @classmethod
    def list_names(cls) -> list:
        """List all indicator names"""
        return list(cls._indicators.keys())

    @classmethod
    def clear(cls):
        """Clear all registered indicators"""
        cls._indicators.clear()

    @classmethod
    def count(cls) -> int:
        """Count registered indicators"""
        return len(cls._indicators)
