# core/microstructure/__init__.py
"""
Market Microstructure Analysis Package

Advanced statistical methods for detecting:
- Price exhaustion
- Reversal points
- Market regime changes
- Structural breaks

WITHOUT relying on traditional indicators with fixed parameters

Main Components:
1. ChangePointDetector - Detect structural breaks (CUSUM, Bayesian, Z-score)
2. HurstExponentAnalyzer - Measure trend persistence vs mean reversion
3. VolumeExhaustionAnalyzer - Detect exhaustion via volume divergence
4. StatisticalSwingDetector - Find statistically significant swing points
5. EntropyAnalyzer - Measure market chaos/order
6. MicrostructureAnalyzer - Unified analyzer combining all methods

Usage:
    from core.microstructure import MicrostructureAnalyzer

    analyzer = MicrostructureAnalyzer()
    signal = analyzer.analyze(data, current_index, fvg_info)

    if signal and signal.confidence > 0.7:
        print(f"Entry signal: {signal.direction} at {signal.price}")
"""

from .change_point_detector import ChangePointDetector
from .hurst_exponent import HurstExponentAnalyzer
from .volume_exhaustion import VolumeExhaustionAnalyzer
from .statistical_swings import StatisticalSwingDetector, SwingPoint
from .entropy_analyzer import EntropyAnalyzer
from .microstructure_analyzer import MicrostructureAnalyzer, MicrostructureSignal

__all__ = [
    'ChangePointDetector',
    'HurstExponentAnalyzer',
    'VolumeExhaustionAnalyzer',
    'StatisticalSwingDetector',
    'SwingPoint',
    'EntropyAnalyzer',
    'MicrostructureAnalyzer',
    'MicrostructureSignal'
]
