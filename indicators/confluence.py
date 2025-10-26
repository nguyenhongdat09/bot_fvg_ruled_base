# indicators/confluence.py
"""
Confluence Scoring System

Combine FVG + Indicators to generate buy/sell scores.

Score >= 70% → High confidence signal
Score >= 60% → Medium confidence signal
Score < 60% → Skip

ENHANCED VERSION (2025-10-26):
- Statistical indicators replace basic indicators
- Volume Profile + POC replaces VWAP
- Skewness/Kurtosis replace Volume Spike
- OBV Divergence replaces basic OBV
- Market Regime Filter (ATR Percentile)

Author: Claude Code
Date: 2025-10-24 (Enhanced: 2025-10-26)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .volatility import ATRIndicator
from .volume import VWAPIndicator, OBVIndicator, VolumeAnalyzer
from .trend import ADXIndicator

# Import statistical indicators
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.indicators.statistical_indicators import StatisticalIndicators, StatisticalScoring


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
                 adx_enabled: bool = True, adx_threshold: float = 25.0,
                 use_statistical: bool = True):
        """
        Initialize Confluence Scorer

        Args:
            weights: Custom weights for each component
                     Basic mode: {'fvg': 50, 'vwap': 20, 'obv': 15, 'volume': 15}
                     Statistical mode: {'fvg': 50, 'poc': 20, 'skewness': 15,
                                       'kurtosis': 10, 'obv_div': 15, 'regime': -10}
            adx_enabled: Enable ADX filter
            adx_threshold: ADX threshold for trending market
            use_statistical: Use statistical indicators (True) or basic indicators (False)
        """
        self.use_statistical = use_statistical

        # Default weights based on mode
        if use_statistical:
            # Statistical mode weights
            self.weights = weights or {
                'fvg': 50,          # Primary signal (unchanged)
                'poc': 20,          # Volume Profile POC (replaces VWAP)
                'skewness': 15,     # Distribution bias (replaces volume spike)
                'kurtosis': 10,     # Fat tails detection (new)
                'obv_div': 15,      # OBV Divergence (replaces basic OBV)
                'regime': -10,      # Market Regime penalty (negative weight!)
            }
        else:
            # Basic mode weights (backward compatible)
            self.weights = weights or {
                'fvg': 50,      # Primary signal
                'vwap': 20,     # Volume confirmation
                'obv': 15,      # Volume trend
                'volume': 15,   # Spike confirmation
            }

        self.adx_enabled = adx_enabled
        self.adx_threshold = adx_threshold

        # Initialize indicators based on mode
        if use_statistical:
            # Statistical indicators are calculated in data preprocessing
            # No need to initialize them here
            self.stat_indicators = StatisticalIndicators()
            self.stat_scoring = StatisticalScoring()
        else:
            # Basic indicators
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

        # === Score components based on mode ===
        if self.use_statistical:
            # Statistical mode
            # POC Score
            poc_score = self._score_poc_statistical(data, index, fvg_signal, fvg_structure.get('bias'))
            result['components']['poc'] = poc_score
            result['total_score'] += poc_score

            # Skewness Score
            skew_score = self._score_skewness_statistical(data, index, fvg_structure.get('bias'))
            result['components']['skewness'] = skew_score
            result['total_score'] += skew_score

            # Kurtosis Score
            kurt_score = self._score_kurtosis_statistical(data, index)
            result['components']['kurtosis'] = kurt_score
            result['total_score'] += kurt_score

            # OBV Divergence Score
            obv_div_score = self._score_obv_divergence_statistical(data, index, fvg_structure.get('bias'))
            result['components']['obv_div'] = obv_div_score
            result['total_score'] += obv_div_score

            # Market Regime Filter (PENALTY for bad conditions)
            regime_penalty = self._score_market_regime_statistical(data, index)
            result['components']['regime'] = regime_penalty
            result['total_score'] += regime_penalty  # Negative penalty!
        else:
            # Basic mode (backward compatible)
            # VWAP Score
            vwap_score = self._score_vwap(data, index, fvg_signal)
            result['components']['vwap'] = vwap_score
            result['total_score'] += vwap_score

            # OBV Score
            obv_score = self._score_obv(data, index, fvg_signal)
            result['components']['obv'] = obv_score
            result['total_score'] += obv_score

            # Volume Score
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

    # ==================== STATISTICAL SCORING METHODS ====================

    def _score_poc_statistical(self, data: pd.DataFrame, index: int,
                               signal: str, fvg_bias: str) -> float:
        """
        Score Volume Profile POC alignment

        Args:
            data: Data with 'POC' and 'POC_distance' columns
            index: Current index
            signal: 'BUY' or 'SELL'
            fvg_bias: 'BULLISH_BIAS' or 'BEARISH_BIAS'

        Returns:
            float: POC score (0-20)
        """
        if 'POC_distance' not in data.columns or pd.isna(data.iloc[index]['POC_distance']):
            return 0

        poc_distance = data.iloc[index]['POC_distance']

        # Use statistical scoring
        raw_score = self.stat_scoring.score_poc_alignment(poc_distance, fvg_bias)

        # Scale to weight
        return (raw_score / 100) * self.weights['poc']

    def _score_skewness_statistical(self, data: pd.DataFrame, index: int,
                                    fvg_bias: str) -> float:
        """
        Score returns skewness

        Args:
            data: Data with 'skewness' column
            index: Current index
            fvg_bias: 'BULLISH_BIAS' or 'BEARISH_BIAS'

        Returns:
            float: Skewness score (0-15)
        """
        if 'skewness' not in data.columns or pd.isna(data.iloc[index]['skewness']):
            return 0

        skewness = data.iloc[index]['skewness']

        # Use statistical scoring
        raw_score = self.stat_scoring.score_skewness(skewness, fvg_bias)

        # Scale to weight
        return (raw_score / 100) * self.weights['skewness']

    def _score_kurtosis_statistical(self, data: pd.DataFrame, index: int) -> float:
        """
        Score returns kurtosis (fat tails)

        Args:
            data: Data with 'kurtosis' column
            index: Current index

        Returns:
            float: Kurtosis score (0-10)
        """
        if 'kurtosis' not in data.columns or pd.isna(data.iloc[index]['kurtosis']):
            return 0

        kurtosis = data.iloc[index]['kurtosis']

        # Use statistical scoring
        raw_score = self.stat_scoring.score_kurtosis(kurtosis)

        # Scale to weight
        return (raw_score / 100) * self.weights['kurtosis']

    def _score_obv_divergence_statistical(self, data: pd.DataFrame, index: int,
                                          fvg_bias: str) -> float:
        """
        Score OBV divergence

        Args:
            data: Data with 'OBV_divergence' column
            index: Current index
            fvg_bias: 'BULLISH_BIAS' or 'BEARISH_BIAS'

        Returns:
            float: OBV divergence score (0-15)
        """
        if 'OBV_divergence' not in data.columns or pd.isna(data.iloc[index]['OBV_divergence']):
            return 0

        divergence = data.iloc[index]['OBV_divergence']

        # Use statistical scoring
        raw_score = self.stat_scoring.score_obv_divergence(int(divergence), fvg_bias)

        # Scale to weight
        return (raw_score / 100) * self.weights['obv_div']

    def _score_market_regime_statistical(self, data: pd.DataFrame, index: int) -> float:
        """
        Score market regime (ATR percentile)

        IMPORTANT: This is a PENALTY system!
        - Good regime (30-70 percentile) = 0 penalty
        - Bad regime (<30 or >70) = negative score

        Args:
            data: Data with 'ATR_percentile' column
            index: Current index

        Returns:
            float: Regime penalty (0 to -10)
        """
        if 'ATR_percentile' not in data.columns or pd.isna(data.iloc[index]['ATR_percentile']):
            return 0  # No penalty if no data

        atr_percentile = data.iloc[index]['ATR_percentile']

        # Use statistical scoring (returns 0-100)
        raw_score = self.stat_scoring.score_market_regime(atr_percentile)

        # Convert to penalty
        # Good regime (score 70-100) = 0 penalty
        # Bad regime (score 0-30) = full penalty
        if raw_score >= 70:
            return 0  # No penalty (good regime)
        elif raw_score >= 50:
            # Moderate penalty
            penalty_ratio = (70 - raw_score) / 20  # 0 to 1
            return self.weights['regime'] * penalty_ratio  # -10 to 0
        else:
            # Full penalty (bad regime)
            return self.weights['regime']  # -10

    # ==================== END STATISTICAL SCORING METHODS ====================

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
