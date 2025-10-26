# core/microstructure/microstructure_analyzer.py
"""
Unified Microstructure Analyzer - Kết hợp tất cả phương pháp phân tích

Main Class kết hợp:
1. Change Point Detection (CUSUM, Bayesian, Z-score)
2. Hurst Exponent (trend persistence)
3. Volume Exhaustion Analysis
4. Statistical Swing Detection
5. Entropy Analysis

Purpose: PHát hiện exhaustion và reversal points sau khi FVG được xác định

Workflow:
1. FVG detected → Price moves away
2. Microstructure analysis → Detect exhaustion signals
3. Statistical swing formed → Entry trigger
4. Return to FVG zone

Author: Claude Code
Date: 2025-10-26
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .change_point_detector import ChangePointDetector
from .hurst_exponent import HurstExponentAnalyzer
from .volume_exhaustion import VolumeExhaustionAnalyzer
from .statistical_swings import StatisticalSwingDetector, SwingPoint
from .entropy_analyzer import EntropyAnalyzer


@dataclass
class MicrostructureSignal:
    """
    Unified Microstructure Signal

    Attributes:
        timestamp: Signal timestamp
        index: Index in dataframe
        signal_type: 'EXHAUSTION', 'REVERSAL', 'SWING', 'CHANGE_POINT'
        direction: 'BULLISH' or 'BEARISH'
        confidence: Confidence score (0-1)
        price: Price at signal
        components: Dict of component scores
        swing_point: SwingPoint object (if applicable)
    """
    timestamp: pd.Timestamp
    index: int
    signal_type: str
    direction: str
    confidence: float
    price: float
    components: Dict = field(default_factory=dict)
    swing_point: Optional[SwingPoint] = None


class MicrostructureAnalyzer:
    """
    Unified Microstructure Analyzer

    Combines all microstructure analysis methods for comprehensive
    market analysis without relying on traditional indicators

    Usage:
        analyzer = MicrostructureAnalyzer()

        # Analyze at specific index
        signal = analyzer.analyze(data, index, fvg_info)

        # Get entry recommendations
        if signal and signal.confidence > 0.7:
            execute_trade(signal.direction, signal.price)
    """

    def __init__(self,
                 # Change point params
                 cpd_method: str = 'cusum',
                 cpd_sensitivity: float = 2.0,

                 # Hurst params
                 hurst_window: int = 100,

                 # Volume exhaustion params
                 vol_ma_period: int = 20,

                 # Swing detection params
                 swing_fractal_period: int = 5,
                 swing_zscore_threshold: float = 1.5,

                 # Entropy params
                 entropy_window: int = 20,
                 entropy_order: int = 3):
        """
        Initialize Microstructure Analyzer

        Args:
            cpd_method: Change point detection method
            cpd_sensitivity: CPD sensitivity
            hurst_window: Window for Hurst calculation
            vol_ma_period: Volume MA period
            swing_fractal_period: Swing fractal period
            swing_zscore_threshold: Swing z-score threshold
            entropy_window: Entropy calculation window
            entropy_order: Permutation entropy order
        """
        # Initialize all analyzers
        self.cpd = ChangePointDetector(
            method=cpd_method,
            sensitivity=cpd_sensitivity
        )

        self.hurst = HurstExponentAnalyzer(
            window_size=hurst_window
        )

        self.volume_exhaustion = VolumeExhaustionAnalyzer(
            volume_ma_period=vol_ma_period
        )

        self.swing_detector = StatisticalSwingDetector(
            fractal_period=swing_fractal_period,
            zscore_threshold=swing_zscore_threshold
        )

        self.entropy = EntropyAnalyzer(
            entropy_window=entropy_window,
            permutation_order=entropy_order
        )

        # Cache for performance
        self._cache = {}

    def analyze(self, data: pd.DataFrame, index: int,
               fvg_info: Optional[Dict] = None,
               atr_series: Optional[pd.Series] = None) -> Optional[MicrostructureSignal]:
        """
        Comprehensive microstructure analysis at given index

        Args:
            data: DataFrame with OHLCV (up to index)
            index: Current index to analyze
            fvg_info: FVG structure info (from FVGManager)
            atr_series: Pre-calculated ATR series

        Returns:
            MicrostructureSignal or None
        """
        if index < 50:  # Need sufficient history
            return None

        # Calculate all components
        components = {}

        # 1. Change Point Detection (20%)
        cpd_signal = self._analyze_change_point(data, index)
        components['change_point'] = cpd_signal

        # 2. Hurst Exponent (20%)
        hurst_signal = self._analyze_hurst(data, index)
        components['hurst'] = hurst_signal

        # 3. Volume Exhaustion (25%)
        exhaustion_signal = self._analyze_volume_exhaustion(data, index)
        components['volume_exhaustion'] = exhaustion_signal

        # 4. Statistical Swing (20%)
        swing_signal = self._analyze_swing(data, index, atr_series)
        components['swing'] = swing_signal

        # 5. Entropy (15%)
        entropy_signal = self._analyze_entropy(data, index)
        components['entropy'] = entropy_signal

        # Combine signals
        combined_signal = self._combine_signals(
            components, data, index, fvg_info
        )

        return combined_signal

    def _analyze_change_point(self, data: pd.DataFrame, index: int) -> Dict:
        """Analyze change points"""
        try:
            # Online detection
            price = data.iloc[index]['close']
            timestamp = data.index[index]

            is_cp = self.cpd.detect_online(price, timestamp)

            if is_cp:
                # Determine direction
                if index >= 5:
                    price_change = data.iloc[index]['close'] - data.iloc[index-5]['close']
                    direction = 'BULLISH' if price_change > 0 else 'BEARISH'
                else:
                    direction = 'NEUTRAL'

                return {
                    'detected': True,
                    'direction': direction,
                    'score': 0.8,
                    'type': 'CHANGE_POINT'
                }

            return {'detected': False, 'score': 0.0}

        except Exception as e:
            return {'detected': False, 'score': 0.0, 'error': str(e)}

    def _analyze_hurst(self, data: pd.DataFrame, index: int) -> Dict:
        """Analyze Hurst exponent"""
        try:
            # Calculate Hurst for recent window
            close_prices = data.iloc[:index+1]['close']

            if len(close_prices) < self.hurst.min_window:
                return {'score': 0.0, 'regime': 'UNKNOWN'}

            # Calculate Hurst
            H = self.hurst.calculate_hurst_rs(close_prices.values)

            # Get regime
            regime = self.hurst.get_market_regime(H)

            # Calculate exhaustion score
            exhaustion_score = self.hurst.get_exhaustion_score(H, 0)

            return {
                'hurst_value': H,
                'regime': regime,
                'exhaustion_score': exhaustion_score,
                'score': exhaustion_score
            }

        except Exception as e:
            return {'score': 0.0, 'error': str(e)}

    def _analyze_volume_exhaustion(self, data: pd.DataFrame, index: int) -> Dict:
        """Analyze volume exhaustion"""
        try:
            analysis = self.volume_exhaustion.calculate_exhaustion_score(
                data.iloc[:index+1], index
            )

            return {
                'exhaustion_score': analysis['exhaustion_score'],
                'direction': analysis['direction'],
                'is_exhausted': analysis['is_exhausted'],
                'score': analysis['exhaustion_score']
            }

        except Exception as e:
            return {'score': 0.0, 'error': str(e)}

    def _analyze_swing(self, data: pd.DataFrame, index: int,
                      atr_series: Optional[pd.Series]) -> Dict:
        """Analyze statistical swings"""
        try:
            # Detect swing at current index
            swing = self.swing_detector.detect_swing_at_index(
                data.iloc[:index+1], index, atr_series
            )

            if swing:
                return {
                    'swing_detected': True,
                    'swing_type': swing.swing_type,
                    'swing_price': swing.price,
                    'swing_strength': swing.strength,
                    'swing_object': swing,
                    'score': swing.strength
                }

            return {'swing_detected': False, 'score': 0.0}

        except Exception as e:
            return {'swing_detected': False, 'score': 0.0, 'error': str(e)}

    def _analyze_entropy(self, data: pd.DataFrame, index: int) -> Dict:
        """Analyze entropy"""
        try:
            analysis = self.entropy.analyze_entropy(data.iloc[:index+1], index)

            # High entropy = chaos/exhaustion
            score = analysis['entropy_score']

            return {
                'entropy_score': score,
                'regime': analysis['regime'],
                'is_chaotic': analysis['is_chaotic'],
                'score': score if analysis['is_chaotic'] else 0.0
            }

        except Exception as e:
            return {'score': 0.0, 'error': str(e)}

    def _combine_signals(self, components: Dict, data: pd.DataFrame,
                        index: int, fvg_info: Optional[Dict]) -> Optional[MicrostructureSignal]:
        """
        Combine all component signals into unified signal

        Weighting:
        - Change Point: 20%
        - Hurst: 20%
        - Volume Exhaustion: 25%
        - Swing: 20%
        - Entropy: 15%
        """
        weights = {
            'change_point': 0.20,
            'hurst': 0.20,
            'volume_exhaustion': 0.25,
            'swing': 0.20,
            'entropy': 0.15
        }

        # Calculate weighted score
        total_score = 0.0

        for component_name, weight in weights.items():
            component = components.get(component_name, {})
            score = component.get('score', 0.0)
            total_score += score * weight

        # Threshold for signal generation
        if total_score < 0.5:
            return None

        # Determine direction (majority vote)
        directions = []

        if components['change_point'].get('detected'):
            directions.append(components['change_point'].get('direction'))

        if components['volume_exhaustion'].get('is_exhausted'):
            directions.append(components['volume_exhaustion'].get('direction'))

        if components['swing'].get('swing_detected'):
            swing_type = components['swing'].get('swing_type')
            directions.append('BULLISH' if swing_type == 'LOW' else 'BEARISH')

        # Count directions
        if not directions:
            return None

        bullish_count = directions.count('BULLISH')
        bearish_count = directions.count('BEARISH')

        if bullish_count > bearish_count:
            final_direction = 'BULLISH'
        elif bearish_count > bullish_count:
            final_direction = 'BEARISH'
        else:
            return None  # No consensus

        # Determine signal type
        if components['swing'].get('swing_detected'):
            signal_type = 'SWING'
        elif components['volume_exhaustion'].get('is_exhausted'):
            signal_type = 'EXHAUSTION'
        elif components['change_point'].get('detected'):
            signal_type = 'CHANGE_POINT'
        else:
            signal_type = 'REVERSAL'

        # Create signal
        signal = MicrostructureSignal(
            timestamp=data.index[index],
            index=index,
            signal_type=signal_type,
            direction=final_direction,
            confidence=total_score,
            price=data.iloc[index]['close'],
            components=components,
            swing_point=components['swing'].get('swing_object')
        )

        return signal

    def analyze_series(self, data: pd.DataFrame,
                      atr_series: Optional[pd.Series] = None) -> List[MicrostructureSignal]:
        """
        Analyze entire data series

        Args:
            data: DataFrame with OHLCV
            atr_series: Pre-calculated ATR

        Returns:
            List[MicrostructureSignal]: All detected signals
        """
        signals = []

        for i in range(50, len(data)):
            signal = self.analyze(data.iloc[:i+1], i, atr_series=atr_series)

            if signal:
                signals.append(signal)

        return signals

    def get_comprehensive_analysis(self, data: pd.DataFrame, index: int) -> Dict:
        """
        Get comprehensive analysis report

        Args:
            data: DataFrame with OHLCV
            index: Current index

        Returns:
            dict: Complete analysis report
        """
        signal = self.analyze(data.iloc[:index+1], index)

        # Calculate individual components
        cpd_analysis = self._analyze_change_point(data.iloc[:index+1], index)
        hurst_analysis = self._analyze_hurst(data.iloc[:index+1], index)
        exhaustion_analysis = self._analyze_volume_exhaustion(data.iloc[:index+1], index)
        swing_analysis = self._analyze_swing(data.iloc[:index+1], index, None)
        entropy_analysis = self._analyze_entropy(data.iloc[:index+1], index)

        return {
            'signal': signal,
            'change_point': cpd_analysis,
            'hurst': hurst_analysis,
            'volume_exhaustion': exhaustion_analysis,
            'swing': swing_analysis,
            'entropy': entropy_analysis,
            'timestamp': data.index[index],
            'price': data.iloc[index]['close']
        }
