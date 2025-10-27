"""
Statistical Analyzer for Smart DCA Recovery System

Analyzes historical price movement patterns to answer:
1. How far does price typically move before reversing?
2. How long do trends typically last?
3. Where does price typically revert to?

This provides statistical foundation for intelligent grid placement.

Author: Claude Code
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class PriceExcursionStats:
    """Statistical summary of price excursions"""

    # Distance statistics (in pips)
    percentile_25: float
    percentile_50: float  # Median
    percentile_75: float
    percentile_90: float
    percentile_95: float
    max_excursion: float
    mean_excursion: float
    std_excursion: float

    # Duration statistics (in bars/candles)
    avg_duration_bars: float
    median_duration_bars: float
    max_duration_bars: int

    # Reversion statistics
    reversion_to_breakeven_prob: float  # Probability of returning to entry
    reversion_to_50pct_prob: float      # Probability of 50% retracement
    avg_reversion_pct: float            # Average retracement percentage

    # Sample size
    n_samples: int

    def __str__(self):
        return f"""Price Excursion Statistics (n={self.n_samples}):
  Distance (pips):
    25th percentile: {self.percentile_25:.1f}
    50th percentile: {self.percentile_50:.1f} (median)
    75th percentile: {self.percentile_75:.1f}
    90th percentile: {self.percentile_90:.1f}
    95th percentile: {self.percentile_95:.1f}
    Max: {self.max_excursion:.1f}
    Mean: {self.mean_excursion:.1f} ± {self.std_excursion:.1f}

  Duration (bars):
    Mean: {self.avg_duration_bars:.1f}
    Median: {self.median_duration_bars:.1f}
    Max: {self.max_duration_bars}

  Reversion:
    Return to breakeven: {self.reversion_to_breakeven_prob*100:.1f}%
    50% retracement: {self.reversion_to_50pct_prob*100:.1f}%
    Avg retracement: {self.avg_reversion_pct*100:.1f}%
"""


class StatisticalAnalyzer:
    """
    Analyze historical price movements to understand:
    - How far price moves before reversing (excursion distance)
    - How long trends last (duration)
    - Where price reverts to (reversion levels)

    This analysis is used to:
    1. Place DCA grid levels intelligently (based on percentiles)
    2. Estimate reversal probability (based on current excursion vs historical)
    3. Calculate position sizes (based on expected reversion)

    Usage:
        analyzer = StatisticalAnalyzer(data, pip_value=0.0001)

        # Analyze bullish moves (price going up)
        bull_stats = analyzer.analyze_excursions(direction='bull')

        # Analyze bearish moves (price going down)
        bear_stats = analyzer.analyze_excursions(direction='bear')

        # Get grid levels for current position
        entry_price = 1.2500
        direction = 'sell'  # We sold, so price going up is adverse
        grid_levels = analyzer.get_grid_levels(entry_price, direction, n_levels=5)
    """

    def __init__(self, data: pd.DataFrame, pip_value: float = 0.0001):
        """
        Initialize statistical analyzer

        Args:
            data: OHLC dataframe with DatetimeIndex
            pip_value: Pip value (0.0001 for 5-digit, 0.01 for JPY pairs)
        """
        self.data = data.copy()
        self.pip_value = pip_value

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required):
            raise ValueError(f"Data must contain columns: {required}")

        # Calculate swing highs and lows
        self._detect_swings()

    def _detect_swings(self, window: int = 5):
        """
        Detect swing highs and lows using local extrema

        Args:
            window: Number of bars on each side to compare
        """
        highs = self.data['high']
        lows = self.data['low']

        # Swing high: high is highest in window
        swing_highs = []
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                swing_highs.append(i)

        # Swing low: low is lowest in window
        swing_lows = []
        for i in range(window, len(lows) - window):
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                swing_lows.append(i)

        self.swing_high_indices = swing_highs
        self.swing_low_indices = swing_lows

    def analyze_excursions(self, direction: str = 'bull') -> PriceExcursionStats:
        """
        Analyze price excursions in given direction

        Args:
            direction: 'bull' (upward moves) or 'bear' (downward moves)

        Returns:
            PriceExcursionStats object with statistical summary
        """
        if direction == 'bull':
            # Analyze upward moves (from swing low to next swing high)
            excursions = self._analyze_bull_excursions()
        else:
            # Analyze downward moves (from swing high to next swing low)
            excursions = self._analyze_bear_excursions()

        if len(excursions) == 0:
            raise ValueError(f"No {direction} excursions found in data")

        # Extract statistics
        distances = [exc['distance_pips'] for exc in excursions]
        durations = [exc['duration_bars'] for exc in excursions]
        reversion_pcts = [exc['reversion_pct'] for exc in excursions]

        # Calculate probabilities
        reversion_to_breakeven = sum(1 for exc in excursions if exc['returned_to_entry']) / len(excursions)
        reversion_to_50pct = sum(1 for exc in excursions if exc['reversion_pct'] >= 0.50) / len(excursions)

        return PriceExcursionStats(
            percentile_25=np.percentile(distances, 25),
            percentile_50=np.percentile(distances, 50),
            percentile_75=np.percentile(distances, 75),
            percentile_90=np.percentile(distances, 90),
            percentile_95=np.percentile(distances, 95),
            max_excursion=np.max(distances),
            mean_excursion=np.mean(distances),
            std_excursion=np.std(distances),
            avg_duration_bars=np.mean(durations),
            median_duration_bars=np.median(durations),
            max_duration_bars=np.max(durations),
            reversion_to_breakeven_prob=reversion_to_breakeven,
            reversion_to_50pct_prob=reversion_to_50pct,
            avg_reversion_pct=np.mean(reversion_pcts),
            n_samples=len(excursions)
        )

    def _analyze_bull_excursions(self) -> List[Dict]:
        """Analyze upward price movements"""
        excursions = []

        # For each swing low, find next swing high
        for i, low_idx in enumerate(self.swing_low_indices):
            # Find next swing high after this low
            next_highs = [h for h in self.swing_high_indices if h > low_idx]
            if not next_highs:
                continue
            high_idx = next_highs[0]

            # Entry at swing low
            entry_price = self.data['low'].iloc[low_idx]
            peak_price = self.data['high'].iloc[high_idx]

            # Calculate excursion
            distance = peak_price - entry_price
            distance_pips = distance / self.pip_value
            duration_bars = high_idx - low_idx

            # Check reversion after peak
            reversion_pct = 0.0
            returned_to_entry = False

            # Look ahead after peak (up to 50 bars or next swing low)
            end_idx = min(high_idx + 50, len(self.data))
            if i + 1 < len(self.swing_low_indices):
                end_idx = min(end_idx, self.swing_low_indices[i + 1])

            if end_idx > high_idx:
                # Find lowest low after peak
                lowest_after_peak = self.data['low'].iloc[high_idx:end_idx].min()

                # Calculate retracement
                retracement = peak_price - lowest_after_peak
                reversion_pct = retracement / distance if distance > 0 else 0

                # Check if returned to entry
                if lowest_after_peak <= entry_price:
                    returned_to_entry = True

            excursions.append({
                'entry_idx': low_idx,
                'peak_idx': high_idx,
                'entry_price': entry_price,
                'peak_price': peak_price,
                'distance_pips': distance_pips,
                'duration_bars': duration_bars,
                'reversion_pct': reversion_pct,
                'returned_to_entry': returned_to_entry
            })

        return excursions

    def _analyze_bear_excursions(self) -> List[Dict]:
        """Analyze downward price movements"""
        excursions = []

        # For each swing high, find next swing low
        for i, high_idx in enumerate(self.swing_high_indices):
            # Find next swing low after this high
            next_lows = [l for l in self.swing_low_indices if l > high_idx]
            if not next_lows:
                continue
            low_idx = next_lows[0]

            # Entry at swing high
            entry_price = self.data['high'].iloc[high_idx]
            bottom_price = self.data['low'].iloc[low_idx]

            # Calculate excursion
            distance = entry_price - bottom_price
            distance_pips = distance / self.pip_value
            duration_bars = low_idx - high_idx

            # Check reversion after bottom
            reversion_pct = 0.0
            returned_to_entry = False

            # Look ahead after bottom (up to 50 bars or next swing high)
            end_idx = min(low_idx + 50, len(self.data))
            if i + 1 < len(self.swing_high_indices):
                end_idx = min(end_idx, self.swing_high_indices[i + 1])

            if end_idx > low_idx:
                # Find highest high after bottom
                highest_after_bottom = self.data['high'].iloc[low_idx:end_idx].max()

                # Calculate retracement
                retracement = highest_after_bottom - bottom_price
                reversion_pct = retracement / distance if distance > 0 else 0

                # Check if returned to entry
                if highest_after_bottom >= entry_price:
                    returned_to_entry = True

            excursions.append({
                'entry_idx': high_idx,
                'bottom_idx': low_idx,
                'entry_price': entry_price,
                'bottom_price': bottom_price,
                'distance_pips': distance_pips,
                'duration_bars': duration_bars,
                'reversion_pct': reversion_pct,
                'returned_to_entry': returned_to_entry
            })

        return excursions

    def get_grid_levels(self,
                       entry_price: float,
                       direction: str,
                       n_levels: int = 5,
                       use_percentiles: bool = True) -> List[Dict]:
        """
        Calculate intelligent grid levels for DCA

        Args:
            entry_price: Initial entry price
            direction: Trade direction ('buy' or 'sell')
            n_levels: Number of DCA levels to create
            use_percentiles: Use statistical percentiles (True) or equal spacing (False)

        Returns:
            List of dicts with keys: level, price, distance_pips, percentile
        """
        # Determine excursion direction
        # If we bought, adverse move is downward (bear)
        # If we sold, adverse move is upward (bull)
        excursion_direction = 'bear' if direction.lower() == 'buy' else 'bull'

        # Get statistics
        stats = self.analyze_excursions(excursion_direction)

        grid_levels = []

        if use_percentiles:
            # Use statistical percentiles for intelligent spacing
            # More levels near common reversal zones
            percentiles = np.linspace(0, 95, n_levels)

            for i, pct in enumerate(percentiles):
                distance_pips = np.percentile(
                    [stats.percentile_25, stats.percentile_50, stats.percentile_75,
                     stats.percentile_90, stats.percentile_95],
                    pct / 95 * 100  # Map to 0-100 range
                )

                # Calculate price
                if direction.lower() == 'buy':
                    price = entry_price - (distance_pips * self.pip_value)
                else:
                    price = entry_price + (distance_pips * self.pip_value)

                grid_levels.append({
                    'level': i + 1,
                    'price': price,
                    'distance_pips': distance_pips,
                    'percentile': pct,
                    'probability_reached': 1 - (pct / 100)  # Higher percentile = lower prob
                })
        else:
            # Equal spacing up to 90th percentile
            max_distance = stats.percentile_90

            for i in range(n_levels):
                distance_pips = max_distance * (i + 1) / n_levels

                if direction.lower() == 'buy':
                    price = entry_price - (distance_pips * self.pip_value)
                else:
                    price = entry_price + (distance_pips * self.pip_value)

                grid_levels.append({
                    'level': i + 1,
                    'price': price,
                    'distance_pips': distance_pips,
                    'percentile': None,
                    'probability_reached': None
                })

        return grid_levels

    def estimate_reversal_probability(self,
                                     entry_price: float,
                                     current_price: float,
                                     direction: str) -> float:
        """
        Estimate probability of reversal based on current excursion vs historical

        Args:
            entry_price: Original entry price
            current_price: Current market price
            direction: Trade direction ('buy' or 'sell')

        Returns:
            Estimated reversal probability (0.0 to 1.0)
        """
        # Calculate current excursion
        if direction.lower() == 'buy':
            excursion = entry_price - current_price  # Adverse move is downward
            excursion_direction = 'bear'
        else:
            excursion = current_price - entry_price  # Adverse move is upward
            excursion_direction = 'bull'

        excursion_pips = abs(excursion) / self.pip_value

        # Get historical statistics
        stats = self.analyze_excursions(excursion_direction)

        # Calculate percentile of current excursion
        if excursion_pips <= stats.percentile_50:
            # Below median - low reversal probability
            prob = 0.3 + (excursion_pips / stats.percentile_50) * 0.2  # 0.3 to 0.5
        elif excursion_pips <= stats.percentile_75:
            # Between median and 75th - moderate probability
            pct_in_range = (excursion_pips - stats.percentile_50) / (stats.percentile_75 - stats.percentile_50)
            prob = 0.5 + pct_in_range * 0.2  # 0.5 to 0.7
        elif excursion_pips <= stats.percentile_90:
            # Between 75th and 90th - high probability
            pct_in_range = (excursion_pips - stats.percentile_75) / (stats.percentile_90 - stats.percentile_75)
            prob = 0.7 + pct_in_range * 0.15  # 0.7 to 0.85
        else:
            # Above 90th percentile - very high probability
            prob = 0.85 + min((excursion_pips - stats.percentile_90) / stats.percentile_90 * 0.15, 0.15)  # 0.85 to 1.0

        return min(prob, 1.0)

    def print_analysis(self):
        """Print comprehensive analysis of both directions"""
        print("\n" + "="*80)
        print("STATISTICAL PRICE MOVEMENT ANALYSIS")
        print("="*80)

        print("\n[BULLISH MOVES] (Upward price movements)")
        print("-" * 80)
        bull_stats = self.analyze_excursions('bull')
        print(bull_stats)

        print("\n[BEARISH MOVES] (Downward price movements)")
        print("-" * 80)
        bear_stats = self.analyze_excursions('bear')
        print(bear_stats)

        print("\n[RECOMMENDATIONS]")
        print("-" * 80)
        print(f"For BUY positions (adverse = downward):")
        print(f"  - Place DCA levels at: {bear_stats.percentile_50:.0f}, {bear_stats.percentile_75:.0f}, {bear_stats.percentile_90:.0f} pips below entry")
        print(f"  - Expect reversal probability >70% after {bear_stats.percentile_75:.0f} pips")
        print(f"  - Max protection: {bear_stats.percentile_95:.0f} pips below entry")

        print(f"\nFor SELL positions (adverse = upward):")
        print(f"  - Place DCA levels at: {bull_stats.percentile_50:.0f}, {bull_stats.percentile_75:.0f}, {bull_stats.percentile_90:.0f} pips above entry")
        print(f"  - Expect reversal probability >70% after {bull_stats.percentile_75:.0f} pips")
        print(f"  - Max protection: {bull_stats.percentile_95:.0f} pips above entry")

        print("="*80 + "\n")


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/home/user/bot_fvg_ruled_base')
    from config import DATA_DIR

    print("\n" + "="*80)
    print("STATISTICAL ANALYZER - EXAMPLE")
    print("="*80)

    # Load data
    data_file = DATA_DIR / 'GBPUSD_M15_180days.csv'

    if not data_file.exists():
        print(f"\nError: Data file not found: {data_file}")
        print("Please run: python data/batch_download_mt5_data.py")
        sys.exit(1)

    print(f"\nLoading data from: {data_file}")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Initialize analyzer
    print("\nInitializing Statistical Analyzer...")
    analyzer = StatisticalAnalyzer(data, pip_value=0.0001)

    # Print analysis
    analyzer.print_analysis()

    # Example: Get grid levels for a SELL trade at 1.2500
    print("\n[EXAMPLE] Grid levels for SELL @ 1.2500:")
    print("-" * 80)
    grid_levels = analyzer.get_grid_levels(
        entry_price=1.2500,
        direction='sell',
        n_levels=5
    )

    for level in grid_levels:
        print(f"Level {level['level']}: {level['price']:.5f} "
              f"({level['distance_pips']:.1f} pips, "
              f"P(reached)={level['probability_reached']*100:.0f}%)")

    # Example: Estimate reversal probability
    print("\n[EXAMPLE] Reversal probability estimation:")
    print("-" * 80)
    entry = 1.2500
    for current in [1.2550, 1.2600, 1.2650, 1.2700]:
        prob = analyzer.estimate_reversal_probability(entry, current, 'sell')
        pips = abs(current - entry) / 0.0001
        print(f"Entry: {entry:.4f}, Current: {current:.4f} "
              f"({pips:.0f} pips adverse) -> Reversal prob: {prob*100:.1f}%")

    print("\n" + "="*80)
    print("DONE")
    print("="*80 + "\n")
