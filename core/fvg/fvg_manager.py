# core/fvg/fvg_manager.py
"""
FVG Manager - Fair Value Gap Management & Tracking

Tasks:
- Manage all FVGs in real-time
- Track state (active/touched)
- Determine market structure (bias)
- Export FVG history
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Literal
from .fvg_model import FVG
from .fvg_detector import FVGDetector


class FVGManager:
    """
    FVG Manager - Manage all FVGs in the system

    Attributes:
        lookback_days: FVG valid for N days (default 90)
        detector: FVGDetector instance
        active_bullish_fvgs: List of active bullish FVGs
        active_bearish_fvgs: List of active bearish FVGs
        all_fvgs_history: History of all FVGs (including touched)
    """

    def __init__(self, lookback_days: int = 90, min_gap_atr_ratio: float = 0.3,
                 min_gap_pips: Optional[float] = None):
        """
        Initialize FVG Manager

        Args:
            lookback_days: FVG valid for how many days
            min_gap_atr_ratio: Gap must be >= ATR × ratio
            min_gap_pips: Gap must be >= pips (optional)
        """
        self.lookback_days = lookback_days

        # Initialize detector
        self.detector = FVGDetector(
            min_gap_atr_ratio=min_gap_atr_ratio,
            min_gap_pips=min_gap_pips
        )

        # FVG storage
        self.active_bullish_fvgs: List[FVG] = []
        self.active_bearish_fvgs: List[FVG] = []
        self.all_fvgs_history: List[FVG] = []

        # Statistics
        self.total_bullish_created = 0
        self.total_bearish_created = 0
        self.total_bullish_touched = 0
        self.total_bearish_touched = 0

    def update(self, data: pd.DataFrame, current_index: int, atr: float) -> Optional[FVG]:
        """
        Update FVG Manager at each new candle

        Flow:
        1. Detect new FVG (if any)
        2. Update old FVG states (check touched)
        3. Remove FVGs > lookback_days
        4. Remove touched FVGs

        Args:
            data: DataFrame OHLC (from start to current_index)
            current_index: Current index
            atr: Current ATR value

        Returns:
            New FVG if detected, None otherwise
        """

        current_candle = data.iloc[current_index]
        current_timestamp = current_candle.name

        # ===== STEP 1: Detect new FVG =====
        new_fvg = self.detector.detect_fvg_at_index(data, current_index, atr)

        if new_fvg is not None:
            # Add to active list
            if new_fvg.fvg_type == 'BULLISH':
                self.active_bullish_fvgs.append(new_fvg)
                self.total_bullish_created += 1
            else:
                self.active_bearish_fvgs.append(new_fvg)
                self.total_bearish_created += 1

            # Add to history
            self.all_fvgs_history.append(new_fvg)

        # ===== STEP 2: Update old FVG states =====
        self._update_fvg_states(current_candle, current_index, current_timestamp)

        # ===== STEP 3: Remove expired FVGs (> lookback_days) =====
        self._remove_expired_fvgs(current_timestamp)

        # ===== STEP 4: Remove touched FVGs =====
        self._remove_touched_fvgs()

        return new_fvg

    def _update_fvg_states(self, current_candle: pd.Series, current_index: int,
                          current_timestamp: pd.Timestamp):
        """
        Update all active FVG states (check touched)

        Args:
            current_candle: Current candle
            current_index: Current index
            current_timestamp: Current timestamp
        """

        candle_high = current_candle['high']
        candle_low = current_candle['low']

        # Check bullish FVGs
        for fvg in self.active_bullish_fvgs:
            was_touched = fvg.check_touched(candle_high, candle_low,
                                           current_index, current_timestamp)
            if was_touched:
                self.total_bullish_touched += 1

        # Check bearish FVGs
        for fvg in self.active_bearish_fvgs:
            was_touched = fvg.check_touched(candle_high, candle_low,
                                           current_index, current_timestamp)
            if was_touched:
                self.total_bearish_touched += 1

    def _remove_expired_fvgs(self, current_timestamp: pd.Timestamp):
        """
        Remove FVGs older than 90 days (or lookback_days)

        Args:
            current_timestamp: Current timestamp
        """

        # Filter bullish FVGs
        self.active_bullish_fvgs = [
            fvg for fvg in self.active_bullish_fvgs
            if fvg.get_age_in_days(current_timestamp) <= self.lookback_days
        ]

        # Filter bearish FVGs
        self.active_bearish_fvgs = [
            fvg for fvg in self.active_bearish_fvgs
            if fvg.get_age_in_days(current_timestamp) <= self.lookback_days
        ]

    def _remove_touched_fvgs(self):
        """
        Remove touched FVGs from active list
        """

        # Filter bullish FVGs
        self.active_bullish_fvgs = [
            fvg for fvg in self.active_bullish_fvgs
            if not fvg.is_touched
        ]

        # Filter bearish FVGs
        self.active_bearish_fvgs = [
            fvg for fvg in self.active_bearish_fvgs
            if not fvg.is_touched
        ]

    def get_market_structure(self, current_price: float) -> Dict:
        """
        Analyze FVG structure to determine market bias

        Logic:
        - BULLISH_BIAS: Has FVG below, NO FVG above
        - BEARISH_BIAS: Has FVG above, NO FVG below
        - BOTH_FVG: Has both FVG below and above
        - NO_FVG: No FVG at all

        Args:
            current_price: Current price

        Returns:
            dict: {
                'bias': str,
                'bullish_fvgs_below': List[FVG],
                'bearish_fvgs_above': List[FVG],
                'nearest_bullish_target': FVG or None,
                'nearest_bearish_target': FVG or None,
                'total_active_fvgs': int
            }
        """

        # ===== Filter FVGs by position =====

        # Bullish FVG BELOW current price
        bullish_fvgs_below = [
            fvg for fvg in self.active_bullish_fvgs
            if fvg.is_valid_target(current_price)
        ]

        # Bearish FVG ABOVE current price
        bearish_fvgs_above = [
            fvg for fvg in self.active_bearish_fvgs
            if fvg.is_valid_target(current_price)
        ]

        # ===== Determine bias =====

        has_bullish_below = len(bullish_fvgs_below) > 0
        has_bearish_above = len(bearish_fvgs_above) > 0

        if has_bullish_below and not has_bearish_above:
            bias = 'BULLISH_BIAS'
        elif has_bearish_above and not has_bullish_below:
            bias = 'BEARISH_BIAS'
        elif has_bullish_below and has_bearish_above:
            bias = 'BOTH_FVG'
        else:
            bias = 'NO_FVG'

        # ===== Find nearest FVG =====

        nearest_bullish_target = None
        nearest_bearish_target = None

        if bullish_fvgs_below:
            # Nearest FVG = smallest distance
            nearest_bullish_target = min(
                bullish_fvgs_below,
                key=lambda fvg: fvg.get_distance_to_price(current_price)
            )

        if bearish_fvgs_above:
            nearest_bearish_target = min(
                bearish_fvgs_above,
                key=lambda fvg: fvg.get_distance_to_price(current_price)
            )

        # ===== Return structure =====

        return {
            'bias': bias,
            'bullish_fvgs_below': bullish_fvgs_below,
            'bearish_fvgs_above': bearish_fvgs_above,
            'nearest_bullish_target': nearest_bullish_target,
            'nearest_bearish_target': nearest_bearish_target,
            'total_active_fvgs': len(self.active_bullish_fvgs) + len(self.active_bearish_fvgs)
        }

    def get_all_active_fvgs(self) -> List[FVG]:
        """
        Get all active FVGs

        Returns:
            List[FVG]: All active FVGs
        """
        return self.active_bullish_fvgs + self.active_bearish_fvgs

    def get_statistics(self) -> Dict:
        """
        Get FVG statistics

        Returns:
            dict: Detailed statistics
        """

        total_active = len(self.active_bullish_fvgs) + len(self.active_bearish_fvgs)

        # Calculate touch rate
        bullish_touch_rate = (
            self.total_bullish_touched / self.total_bullish_created * 100
            if self.total_bullish_created > 0 else 0
        )

        bearish_touch_rate = (
            self.total_bearish_touched / self.total_bearish_created * 100
            if self.total_bearish_created > 0 else 0
        )

        return {
            'total_active': total_active,
            'active_bullish': len(self.active_bullish_fvgs),
            'active_bearish': len(self.active_bearish_fvgs),
            'total_bullish_created': self.total_bullish_created,
            'total_bearish_created': self.total_bearish_created,
            'total_bullish_touched': self.total_bullish_touched,
            'total_bearish_touched': self.total_bearish_touched,
            'bullish_touch_rate': round(bullish_touch_rate, 2),
            'bearish_touch_rate': round(bearish_touch_rate, 2),
            'total_fvgs_in_history': len(self.all_fvgs_history)
        }

    def export_history_to_dataframe(self) -> pd.DataFrame:
        """
        Export all FVG history to DataFrame

        Returns:
            pd.DataFrame: FVG history
        """

        if not self.all_fvgs_history:
            return pd.DataFrame()

        records = [fvg.to_dict() for fvg in self.all_fvgs_history]
        df = pd.DataFrame(records)

        return df

    def export_active_to_dataframe(self) -> pd.DataFrame:
        """
        Export active FVGs to DataFrame

        Returns:
            pd.DataFrame: Active FVGs
        """

        active_fvgs = self.get_all_active_fvgs()

        if not active_fvgs:
            return pd.DataFrame()

        records = [fvg.to_dict() for fvg in active_fvgs]
        df = pd.DataFrame(records)

        return df

    def reset(self):
        """
        Reset entire FVG Manager (for testing)
        """
        self.active_bullish_fvgs.clear()
        self.active_bearish_fvgs.clear()
        self.all_fvgs_history.clear()
        self.total_bullish_created = 0
        self.total_bearish_created = 0
        self.total_bullish_touched = 0
        self.total_bearish_touched = 0

    def __repr__(self) -> str:
        """String representation"""
        return (f"FVGManager(active_bullish={len(self.active_bullish_fvgs)}, "
                f"active_bearish={len(self.active_bearish_fvgs)}, "
                f"total_history={len(self.all_fvgs_history)})")


# ===== HELPER FUNCTIONS =====

def validate_signal_with_fvg(fvg_structure: Dict, signal_direction: Literal['BUY', 'SELL']) -> bool:
    """
    Validate trading signal with FVG structure

    GOLDEN RULES:
    - BULLISH_BIAS: Only trade BUY
    - BEARISH_BIAS: Only trade SELL
    - BOTH_FVG: Trade according to indicators
    - NO_FVG: NO TRADE

    Args:
        fvg_structure: Dict from get_market_structure()
        signal_direction: 'BUY' or 'SELL'

    Returns:
        bool: True if signal is valid
    """

    bias = fvg_structure['bias']

    if bias == 'NO_FVG':
        return False  # No FVG target

    if bias == 'BULLISH_BIAS':
        return signal_direction == 'BUY'

    if bias == 'BEARISH_BIAS':
        return signal_direction == 'SELL'

    if bias == 'BOTH_FVG':
        return True  # Trade according to indicators

    return False


def get_fvg_target(fvg_structure: Dict, signal_direction: Literal['BUY', 'SELL']) -> Optional[FVG]:
    """
    Get FVG target for trade

    Args:
        fvg_structure: Dict from get_market_structure()
        signal_direction: 'BUY' or 'SELL'

    Returns:
        FVG target or None
    """

    if signal_direction == 'BUY':
        return fvg_structure.get('nearest_bullish_target')
    elif signal_direction == 'SELL':
        return fvg_structure.get('nearest_bearish_target')

    return None
