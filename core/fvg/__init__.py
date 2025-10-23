# core/fvg/__init__.py
"""
FVG Module - Fair Value Gap Detection & Management

This module provides complete FVG functionality:
- FVG Model: Object definition
- FVG Detector: Gap detection
- FVG Manager: Tracking & management
- FVG Visualizer: Chart visualization
"""

from .fvg_model import (
    FVG,
    generate_fvg_id,
    calculate_fvg_strength
)

from .fvg_detector import FVGDetector

from .fvg_manager import (
    FVGManager,
    validate_signal_with_fvg,
    get_fvg_target
)

from .fvg_visualizer import (
    FVGVisualizer,
    quick_plot_fvgs,
    compare_fvg_periods
)

__all__ = [
    # FVG Model
    'FVG',
    'generate_fvg_id',
    'calculate_fvg_strength',

    # FVG Detector
    'FVGDetector',

    # FVG Manager
    'FVGManager',
    'validate_signal_with_fvg',
    'get_fvg_target',

    # FVG Visualizer
    'FVGVisualizer',
    'quick_plot_fvgs',
    'compare_fvg_periods'
]
