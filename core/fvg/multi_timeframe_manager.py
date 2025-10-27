# core/fvg/multi_timeframe_manager.py
"""
Multi-Timeframe FVG Manager

Manage FVGs across multiple timeframes with automatic resampling and alignment.

Example:
    mtf = MultiTimeframeManager(m15_data, base_timeframe='M15')
    mtf.add_fvg_timeframe('H1')
    mtf.add_fvg_timeframe('H4')

    for i in range(100, len(m15_data)):
        mtf.update(i)
        h1_bias = mtf.get_fvg_bias('H1', i)
        h4_bias = mtf.get_fvg_bias('H4', i)
"""

import pandas as pd
from typing import Dict, Optional, List
from .fvg_manager import FVGManager

# Import ATR indicator
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from indicators.volatility import ATRIndicator


class MultiTimeframeManager:
    """
    Manage FVGs across multiple timeframes

    This class automatically:
    1. Resamples base data to higher timeframes
    2. Maintains separate FVGManager for each timeframe
    3. Aligns higher timeframe states to base timeframe
    4. Ensures no look-ahead bias

    Attributes:
        base_data: Base timeframe data (smallest timeframe)
        base_timeframe: Base timeframe name (e.g., 'M15')
        managers: Dict of FVGManager for each timeframe
        resampled_data: Dict of resampled DataFrames
        states: Dict of aligned states for each timeframe
    """

    # Timeframe mapping for pandas resample
    TIMEFRAME_MAP = {
        'M1': '1T',
        'M5': '5T',
        'M15': '15T',
        'M30': '30T',
        'H1': '1H',
        'H4': '4H',
        'D1': '1D',
        'W1': '1W',
        'MN1': '1M'
    }

    def __init__(self, base_data: pd.DataFrame, base_timeframe: str = 'M15'):
        """
        Initialize MultiTimeframeManager

        Args:
            base_data: DataFrame with OHLCV data (smallest timeframe)
            base_timeframe: Base timeframe name (e.g., 'M15')
        """
        self.base_data = base_data
        self.base_timeframe = base_timeframe

        # Storage for each timeframe
        self.managers: Dict[str, FVGManager] = {}
        self.resampled_data: Dict[str, pd.DataFrame] = {}
        self.states: Dict[str, pd.DataFrame] = {}  # Processed states
        self.last_processed_index: Dict[str, int] = {}  # Track processing progress

        print(f"MultiTimeframeManager initialized")
        print(f"  Base timeframe: {base_timeframe}")
        print(f"  Base data: {len(base_data)} candles")
        print(f"  Date range: {base_data.index[0]} to {base_data.index[-1]}")

    def add_fvg_timeframe(self, timeframe: str, lookback_days: int = 90,
                         min_gap_atr_ratio: float = 0.3):
        """
        Add FVG analysis for a timeframe

        Args:
            timeframe: Timeframe to add (e.g., 'H1', 'H4', 'D1')
            lookback_days: FVG validity period
            min_gap_atr_ratio: Minimum gap size relative to ATR
        """

        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}. "
                           f"Supported: {list(self.TIMEFRAME_MAP.keys())}")

        print(f"\nAdding FVG timeframe: {timeframe}")

        # Step 1: Resample base data to target timeframe
        resampled = self._resample_data(timeframe)
        self.resampled_data[timeframe] = resampled

        print(f"  Resampled data: {len(resampled)} candles")

        # Step 2: Calculate ATR for this timeframe
        atr_indicator = ATRIndicator(period=14)
        resampled['atr'] = atr_indicator.calculate(resampled)

        # Step 3: Create FVGManager for this timeframe
        manager = FVGManager(
            lookback_days=lookback_days,
            min_gap_atr_ratio=min_gap_atr_ratio
        )
        self.managers[timeframe] = manager

        # Step 4: Initialize state storage
        self.states[timeframe] = pd.DataFrame(columns=[
            'timestamp', 'bias', 'total_active_fvgs',
            'active_bullish', 'active_bearish',
            'nearest_bullish_target', 'nearest_bearish_target'
        ])

        self.last_processed_index[timeframe] = -1

        print(f"  FVGManager created for {timeframe}")

    def _resample_data(self, timeframe: str) -> pd.DataFrame:
        """
        Resample base data to target timeframe

        Args:
            timeframe: Target timeframe (e.g., 'H1')

        Returns:
            pd.DataFrame: Resampled OHLCV data
        """

        resample_rule = self.TIMEFRAME_MAP[timeframe]

        resampled = self.base_data.resample(resample_rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Remove rows with NaN (incomplete candles)
        resampled = resampled.dropna()

        return resampled

    def update(self, base_index: int):
        """
        Update all timeframes up to base_index

        This method:
        1. Finds corresponding index in each higher timeframe
        2. Updates FVGManager sequentially (no look-ahead bias)
        3. Stores the state for later alignment

        Args:
            base_index: Current index in base timeframe
        """

        base_time = self.base_data.index[base_index]

        # Update each timeframe
        for tf, manager in self.managers.items():
            resampled = self.resampled_data[tf]

            # Find corresponding index in resampled timeframe
            # Use 'ffill' to get the latest completed candle BEFORE or AT base_time
            tf_indices = resampled.index.get_indexer([base_time], method='ffill')
            tf_idx = tf_indices[0]

            # Skip if no valid timeframe candle yet
            if tf_idx < 0:
                continue

            # Skip if already processed this timeframe candle
            if tf_idx <= self.last_processed_index[tf]:
                continue

            # Update FVGManager sequentially (no look-ahead bias)
            tf_candle = resampled.iloc[tf_idx]
            manager.update(
                resampled.iloc[:tf_idx+1],
                tf_idx,
                tf_candle['atr']
            )

            # Get structure at this timeframe index
            structure = manager.get_market_structure(tf_candle['close'])

            # Store state
            state = {
                'timestamp': resampled.index[tf_idx],
                'bias': structure['bias'],
                'total_active_fvgs': structure['total_active_fvgs'],
                'active_bullish': len(structure['bullish_fvgs_below']),
                'active_bearish': len(structure['bearish_fvgs_above']),
                'nearest_bullish_target': structure['nearest_bullish_target'],
                'nearest_bearish_target': structure['nearest_bearish_target']
            }

            # Append to states DataFrame
            self.states[tf] = pd.concat([
                self.states[tf],
                pd.DataFrame([state])
            ], ignore_index=True)

            # Update last processed index
            self.last_processed_index[tf] = tf_idx

    def get_fvg_bias(self, timeframe: str, base_index: int) -> Optional[str]:
        """
        Get FVG bias for a timeframe at base_index

        Args:
            timeframe: Target timeframe (e.g., 'H1')
            base_index: Index in base timeframe

        Returns:
            str: Bias ('BULLISH_BIAS', 'BEARISH_BIAS', 'BOTH_FVG', 'NO_FVG')
                 or None if no data available
        """

        if timeframe not in self.states:
            raise ValueError(f"Timeframe {timeframe} not added. "
                           f"Use add_fvg_timeframe() first.")

        base_time = self.base_data.index[base_index]
        states_df = self.states[timeframe]

        if len(states_df) == 0:
            return None

        # Find latest state BEFORE or AT base_time (forward fill)
        valid_states = states_df[states_df['timestamp'] <= base_time]

        if len(valid_states) == 0:
            return None

        latest_state = valid_states.iloc[-1]
        return latest_state['bias']

    def get_fvg_structure(self, timeframe: str, base_index: int) -> Optional[Dict]:
        """
        Get full FVG structure for a timeframe at base_index

        Args:
            timeframe: Target timeframe
            base_index: Index in base timeframe

        Returns:
            dict: Full structure or None if no data available
        """

        if timeframe not in self.states:
            raise ValueError(f"Timeframe {timeframe} not added.")

        base_time = self.base_data.index[base_index]
        states_df = self.states[timeframe]

        if len(states_df) == 0:
            return None

        # Find latest state BEFORE or AT base_time
        valid_states = states_df[states_df['timestamp'] <= base_time]

        if len(valid_states) == 0:
            return None

        latest_state = valid_states.iloc[-1]

        return {
            'timestamp': latest_state['timestamp'],
            'bias': latest_state['bias'],
            'total_active_fvgs': latest_state['total_active_fvgs'],
            'active_bullish': latest_state['active_bullish'],
            'active_bearish': latest_state['active_bearish'],
            'nearest_bullish_target': latest_state['nearest_bullish_target'],
            'nearest_bearish_target': latest_state['nearest_bearish_target']
        }

    def get_statistics(self, timeframe: str) -> Dict:
        """
        Get FVG statistics for a timeframe

        Args:
            timeframe: Target timeframe

        Returns:
            dict: Statistics from FVGManager
        """

        if timeframe not in self.managers:
            raise ValueError(f"Timeframe {timeframe} not added.")

        return self.managers[timeframe].get_statistics()

    def get_available_timeframes(self) -> List[str]:
        """
        Get list of available timeframes

        Returns:
            List[str]: List of timeframe names
        """
        return list(self.managers.keys())

    def get_resampled_data(self, timeframe: str) -> pd.DataFrame:
        """
        Get resampled data for a timeframe (for debugging/visualization)

        Args:
            timeframe: Target timeframe

        Returns:
            pd.DataFrame: Resampled OHLCV data
        """

        if timeframe not in self.resampled_data:
            raise ValueError(f"Timeframe {timeframe} not added.")

        return self.resampled_data[timeframe]

    def get_states_dataframe(self, timeframe: str) -> pd.DataFrame:
        """
        Get all processed states as DataFrame (for debugging/analysis)

        Args:
            timeframe: Target timeframe

        Returns:
            pd.DataFrame: All states
        """

        if timeframe not in self.states:
            raise ValueError(f"Timeframe {timeframe} not added.")

        return self.states[timeframe]

    def __repr__(self) -> str:
        """String representation"""
        return (f"MultiTimeframeManager("
                f"base={self.base_timeframe}, "
                f"timeframes={list(self.managers.keys())}, "
                f"candles={len(self.base_data)})")
