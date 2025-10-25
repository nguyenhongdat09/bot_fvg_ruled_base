# indicators/confluence.py
"""
Confluence Scoring System

Combine FVG + Indicators to generate buy/sell scores.

Score >= 70% → High confidence signal
Score >= 60% → Medium confidence signal
Score < 60% → Skip

Author: Claude Code
Date: 2025-10-24
"""

import pandas as pd
from typing import Dict, Any, Optional
from .volatility import ATRIndicator
from .volume import VWAPIndicator, OBVIndicator, VolumeAnalyzer
from .trend import ADXIndicator


class ConfluenceScorer:
    """
    Confluence Scoring System

    Combines multiple indicators to score trading signals.

    Architecture:
    - Extensible: Easy to add/remove indicators
    - Config-driven: Weights configurable
    - Transparent: Returns detailed breakdown

    Default Scoring:
    - FVG: 50 points (primary signal)
    - VWAP: 20 points (volume confirmation)
    - OBV: 15 points (volume trend)
    - Volume Spike: 15 points (confirmation)
    - ADX Filter: Enable/disable based on trend strength

    Total: 100 points

    Usage:
        scorer = ConfluenceScorer()

        # Calculate score at index i
        score = scorer.calculate_score(
            data=data,
            index=i,
            fvg_structure=fvg_structure,
            atr_value=atr_value
        )

        if score['total_score'] >= 70:
            execute_trade(score['signal'], score['confidence'])
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 adx_enabled: bool = True, adx_threshold: float = 25.0):
        """
        Initialize Confluence Scorer

        Args:
            weights: Custom weights for each component
                     Default: {'fvg': 50, 'vwap': 20, 'obv': 15, 'volume': 15}
            adx_enabled: Enable ADX filter
            adx_threshold: ADX threshold for trending market
        """
        # Default weights
        self.weights = weights or {
            'fvg': 50,      # Primary signal
            'vwap': 20,     # Volume confirmation
            'obv': 15,      # Volume trend
            'volume': 15,   # Spike confirmation
        }

        self.adx_enabled = adx_enabled
        self.adx_threshold = adx_threshold

        # Initialize indicators
        self.vwap = VWAPIndicator()
        self.obv = OBVIndicator()
        self.volume_analyzer = VolumeAnalyzer(period=20, spike_threshold=1.5)
        self.adx = ADXIndicator(period=14) if adx_enabled else None

        # Validate weights
        self._validate_weights()

    def _validate_weights(self):
        """Validate that weights sum to 100"""
        total = sum(self.weights.values())
        if abs(total - 100) > 0.01:  # Allow small floating point error
            raise ValueError(f"Weights must sum to 100, got {total}")

    def calculate_score(self, data: pd.DataFrame, index: int,
                       fvg_structure: Dict[str, Any],
                       atr_value: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate confluence score at given index

        Args:
            data: OHLCV DataFrame (up to index)
            index: Current index to score
            fvg_structure: FVG structure from FVGManager.get_market_structure()
                          Must contain: 'bias', 'total_active_fvgs', etc.
            atr_value: Optional ATR value (for SL/TP calculation)

        Returns:
            dict: {
                'total_score': float (0-100),
                'signal': str ('BUY', 'SELL', 'NEUTRAL'),
                'confidence': str ('HIGH', 'MEDIUM', 'LOW'),
                'components': dict (individual scores),
                'should_trade': bool,
                'reason': str (explanation),
                'sl_tp': dict (if atr_value provided)
            }
        """
        # Initialize result
        result = {
            'total_score': 0,
            'signal': 'NEUTRAL',
            'confidence': 'LOW',
            'components': {},
            'should_trade': False,
            'reason': '',
            'sl_tp': None
        }

        # === FVG Score (Primary Signal) ===
        fvg_score, fvg_signal = self._score_fvg(fvg_structure)
        result['components']['fvg'] = fvg_score
        result['signal'] = fvg_signal

        if fvg_score == 0:
            result['reason'] = 'No FVG bias'
            return result

        result['total_score'] += fvg_score

        # === ADX Filter (if enabled) ===
        if self.adx_enabled:
            adx_filter = self._check_adx_filter(data, index)
            result['components']['adx_filter'] = adx_filter

            if not adx_filter:
                result['reason'] = f'Ranging market (ADX < {self.adx_threshold})'
                result['should_trade'] = False
                return result

        # === VWAP Score ===
        vwap_score = self._score_vwap(data, index, fvg_signal)
        result['components']['vwap'] = vwap_score
        result['total_score'] += vwap_score

        # === OBV Score ===
        obv_score = self._score_obv(data, index, fvg_signal)
        result['components']['obv'] = obv_score
        result['total_score'] += obv_score

        # === Volume Score ===
        volume_score = self._score_volume(data, index)
        result['components']['volume'] = volume_score
        result['total_score'] += volume_score

        # === Determine confidence ===
        if result['total_score'] >= 70:
            result['confidence'] = 'HIGH'
            result['should_trade'] = True
            result['reason'] = f"High confluence ({result['total_score']:.0f}%)"
        elif result['total_score'] >= 60:
            result['confidence'] = 'MEDIUM'
            result['should_trade'] = True
            result['reason'] = f"Medium confluence ({result['total_score']:.0f}%)"
        else:
            result['confidence'] = 'LOW'
            result['should_trade'] = False
            result['reason'] = f"Low confluence ({result['total_score']:.0f}%)"

        # === Calculate SL/TP if ATR provided ===
        if atr_value and result['should_trade']:
            entry_price = data.iloc[index]['close']
            atr_indicator = ATRIndicator()
            result['sl_tp'] = atr_indicator.get_sl_tp_levels(
                entry_price, atr_value, result['signal']
            )

        return result

    def _score_fvg(self, fvg_structure: Dict[str, Any]) -> tuple:
        """
        Score FVG (Primary Signal)

        Args:
            fvg_structure: FVG structure

        Returns:
            tuple: (score, signal)
        """
        bias = fvg_structure.get('bias', 'NO_FVG')

        if bias == 'BULLISH_BIAS':
            return (self.weights['fvg'], 'BUY')
        elif bias == 'BEARISH_BIAS':
            return (self.weights['fvg'], 'SELL')
        else:
            return (0, 'NEUTRAL')

    def _score_vwap(self, data: pd.DataFrame, index: int, signal: str) -> float:
        """
        Score VWAP confirmation

        Args:
            data: OHLCV data
            index: Current index
            signal: 'BUY' or 'SELL'

        Returns:
            float: VWAP score (0-20)
        """
        if signal == 'NEUTRAL':
            return 0

        # Calculate VWAP
        vwap_values = self.vwap.calculate(data)
        current_vwap = vwap_values.iloc[index]
        current_price = data.iloc[index]['close']

        # BUY: price > VWAP (institutional bullish)
        if signal == 'BUY' and current_price > current_vwap:
            return self.weights['vwap']

        # SELL: price < VWAP (institutional bearish)
        if signal == 'SELL' and current_price < current_vwap:
            return self.weights['vwap']

        return 0

    def _score_obv(self, data: pd.DataFrame, index: int, signal: str) -> float:
        """
        Score OBV trend

        Args:
            data: OHLCV data
            index: Current index
            signal: 'BUY' or 'SELL'

        Returns:
            float: OBV score (0-15)
        """
        if signal == 'NEUTRAL' or index < 20:
            return 0

        # Calculate OBV with SMA
        obv_data = self.obv.get_obv_sma(data, period=20)
        current_obv = obv_data.iloc[index]['obv']
        current_obv_sma = obv_data.iloc[index]['obv_sma']

        # BUY: OBV > SMA (accumulation)
        if signal == 'BUY' and current_obv > current_obv_sma:
            return self.weights['obv']

        # SELL: OBV < SMA (distribution)
        if signal == 'SELL' and current_obv < current_obv_sma:
            return self.weights['obv']

        return 0

    def _score_volume(self, data: pd.DataFrame, index: int) -> float:
        """
        Score volume spike

        Args:
            data: OHLCV data
            index: Current index

        Returns:
            float: Volume score (0-15)
        """
        if index < 20:
            return 0

        # Calculate volume analysis
        vol_analysis = self.volume_analyzer.calculate(data)
        volume_ratio = vol_analysis.iloc[index]['volume_ratio']

        # Map ratio to score
        if pd.isna(volume_ratio):
            return 0

        if volume_ratio > 2.0:
            return self.weights['volume']  # Very strong spike (15 points)
        elif volume_ratio > 1.5:
            return self.weights['volume'] * 0.7  # Strong spike (10 points)
        elif volume_ratio > 1.2:
            return self.weights['volume'] * 0.3  # Moderate (5 points)
        else:
            return 0

    def _check_adx_filter(self, data: pd.DataFrame, index: int) -> bool:
        """
        Check ADX filter

        Args:
            data: OHLCV data
            index: Current index

        Returns:
            bool: True if trending (should trade), False if ranging (skip)
        """
        if not self.adx_enabled or index < 14:
            return True  # Don't filter if disabled or not enough data

        adx_data = self.adx.calculate(data)
        current_adx = adx_data.iloc[index]['adx']

        return current_adx >= self.adx_threshold

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update scoring weights

        Args:
            new_weights: New weights dict

        Example:
            scorer.update_weights({
                'fvg': 60,      # Increase FVG weight
                'vwap': 20,
                'obv': 10,
                'volume': 10
            })
        """
        self.weights = new_weights
        self._validate_weights()

    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return self.weights.copy()

    def enable_adx_filter(self, threshold: float = 25.0):
        """
        Enable ADX filter

        Args:
            threshold: ADX threshold
        """
        self.adx_enabled = True
        self.adx_threshold = threshold
        if self.adx is None:
            self.adx = ADXIndicator(period=14)

    def disable_adx_filter(self):
        """Disable ADX filter"""
        self.adx_enabled = False
