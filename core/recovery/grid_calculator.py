"""
Grid Calculator for Smart DCA Recovery System

Calculates intelligent grid spacing for DCA orders based on:
1. Statistical analysis (price excursion percentiles)
2. Current volatility (ATR)
3. Reversal predictor signals
4. Risk management constraints

Author: Claude Code
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class GridLevel:
    """Single DCA grid level"""
    level: int              # Grid level number (1, 2, 3, ...)
    price: float            # Price to place order
    distance_pips: float    # Distance from entry in pips
    lot_size: float         # Lot size for this level
    total_lot: float        # Cumulative lot size after this level
    avg_price: float        # Average entry price after this level
    breakeven_price: float  # Price needed to breakeven (incl. commission)
    required_reversion_pips: float  # Pips needed to reach breakeven
    reversal_probability: float     # Estimated probability of reaching this level
    risk_usd: float         # Risk in USD at this level

    def __str__(self):
        return (f"Level {self.level}: {self.price:.5f} "
                f"({self.distance_pips:+.0f} pips, lot={self.lot_size:.2f})")


class GridCalculator:
    """
    Calculate intelligent DCA grid for recovery

    Features:
    - Statistical-based spacing (uses historical percentiles)
    - ATR-adjusted for current volatility
    - Progressive lot sizing (martingale or fixed ratio)
    - Risk-aware (max drawdown limits)
    - Breakeven calculation including commissions

    Usage:
        calculator = GridCalculator(
            statistical_analyzer=stat_analyzer,
            reversal_predictor=rev_predictor,
            pip_value=0.0001,
            pip_value_in_usd=10.0,
            commission_per_lot=7.0
        )

        # Calculate grid for a position
        grid = calculator.calculate_grid(
            entry_price=1.2500,
            entry_lot=0.1,
            direction='sell',
            current_idx=1000,
            n_levels=5,
            lot_multiplier=1.5,
            max_risk_usd=500
        )

        # Print grid
        for level in grid:
            print(level)
    """

    def __init__(self,
                 statistical_analyzer,
                 reversal_predictor,
                 pip_value: float = 0.0001,
                 pip_value_in_usd: float = 10.0,
                 commission_per_lot: float = 7.0):
        """
        Initialize grid calculator

        Args:
            statistical_analyzer: StatisticalAnalyzer instance
            reversal_predictor: ReversalPredictor instance
            pip_value: Pip value (0.0001 for 5-digit, 0.01 for JPY)
            pip_value_in_usd: USD per pip per standard lot
            commission_per_lot: Commission per lot (round trip)
        """
        self.stat_analyzer = statistical_analyzer
        self.rev_predictor = reversal_predictor
        self.pip_value = pip_value
        self.pip_value_in_usd = pip_value_in_usd
        self.commission_per_lot = commission_per_lot

    def calculate_grid(self,
                      entry_price: float,
                      entry_lot: float,
                      direction: str,
                      current_idx: int,
                      n_levels: int = 5,
                      lot_multiplier: float = 1.5,
                      max_risk_usd: Optional[float] = None,
                      spacing_mode: str = 'statistical') -> List[GridLevel]:
        """
        Calculate DCA grid levels

        Args:
            entry_price: Initial entry price
            entry_lot: Initial lot size
            direction: Trade direction ('buy' or 'sell')
            current_idx: Current index in data (for volatility/prediction)
            n_levels: Number of DCA levels to create
            lot_multiplier: Lot size multiplier for each level (e.g., 1.5 = +50% each level)
            max_risk_usd: Maximum risk in USD (optional, will limit grid depth)
            spacing_mode: 'statistical' (percentile-based) or 'atr' (ATR-based) or 'hybrid'

        Returns:
            List of GridLevel objects
        """
        direction = direction.lower()
        if direction not in ['buy', 'sell']:
            raise ValueError(f"Direction must be 'buy' or 'sell', got: {direction}")

        # Get statistical info
        excursion_direction = 'bear' if direction == 'buy' else 'bull'
        stats = self.stat_analyzer.analyze_excursions(excursion_direction)

        # Get current ATR for volatility adjustment
        if hasattr(self.stat_analyzer, 'data') and 'atr' in self.stat_analyzer.data.columns:
            current_atr = self.stat_analyzer.data['atr'].iloc[current_idx] if current_idx < len(self.stat_analyzer.data) else stats.mean_excursion * self.pip_value
        else:
            # Fallback to statistical mean if ATR not available
            current_atr = stats.mean_excursion * self.pip_value

        # Calculate grid spacing
        if spacing_mode == 'statistical':
            # Use percentiles directly
            spacing_pips = [
                stats.percentile_50,
                stats.percentile_75,
                stats.percentile_90,
                stats.percentile_95,
            ]
            # Extend if needed
            while len(spacing_pips) < n_levels:
                last = spacing_pips[-1]
                spacing_pips.append(last * 1.2)  # 20% further each time

        elif spacing_mode == 'atr':
            # Use ATR multiples
            atr_pips = current_atr / self.pip_value
            spacing_pips = [atr_pips * (i + 1) * 2 for i in range(n_levels)]

        else:  # hybrid
            # Combine statistical and ATR
            atr_pips = current_atr / self.pip_value
            base_spacing = [stats.percentile_50, stats.percentile_75, stats.percentile_90]

            # Adjust by current ATR vs average
            avg_atr_pips = stats.mean_excursion
            atr_ratio = atr_pips / avg_atr_pips if avg_atr_pips > 0 else 1.0

            spacing_pips = [s * atr_ratio for s in base_spacing]

            # Extend if needed
            while len(spacing_pips) < n_levels:
                spacing_pips.append(spacing_pips[-1] * 1.2)

        # Only use first n_levels
        spacing_pips = spacing_pips[:n_levels]

        # Calculate grid levels
        grid = []
        cumulative_lot = entry_lot
        cumulative_cost = entry_price * entry_lot  # Price × lot

        for i, distance_pips in enumerate(spacing_pips):
            level_num = i + 1

            # Calculate price
            if direction == 'buy':
                level_price = entry_price - (distance_pips * self.pip_value)
            else:
                level_price = entry_price + (distance_pips * self.pip_value)

            # Calculate lot size
            if lot_multiplier == 1.0:
                level_lot = entry_lot  # Fixed lot size
            else:
                level_lot = entry_lot * (lot_multiplier ** level_num)

            # Round lot size
            level_lot = round(level_lot, 2)

            # Update cumulative
            cumulative_lot += level_lot
            cumulative_cost += level_price * level_lot

            # Calculate average entry price
            avg_price = cumulative_cost / cumulative_lot

            # Calculate breakeven (including commission)
            total_commission = cumulative_lot * self.commission_per_lot

            if direction == 'buy':
                # Need to go above avg to profit
                breakeven_pips = (total_commission / cumulative_lot) / self.pip_value_in_usd
                breakeven_price = avg_price + (breakeven_pips * self.pip_value)
                required_reversion = breakeven_price - level_price
            else:
                # Need to go below avg to profit
                breakeven_pips = (total_commission / cumulative_lot) / self.pip_value_in_usd
                breakeven_price = avg_price - (breakeven_pips * self.pip_value)
                required_reversion = level_price - breakeven_price

            required_reversion_pips = abs(required_reversion) / self.pip_value

            # Estimate reversal probability using predictor
            try:
                signal = self.rev_predictor.predict(
                    current_idx=current_idx,
                    entry_price=entry_price,
                    current_price=level_price,
                    direction=direction
                )
                reversal_prob = signal.probability
            except:
                # Fallback to statistical estimate
                reversal_prob = self.stat_analyzer.estimate_reversal_probability(
                    entry_price, level_price, direction
                )

            # Calculate risk (unrealized loss at this level)
            if direction == 'buy':
                unrealized_loss = (avg_price - level_price) * cumulative_lot / self.pip_value * self.pip_value_in_usd
            else:
                unrealized_loss = (level_price - avg_price) * cumulative_lot / self.pip_value * self.pip_value_in_usd

            risk_usd = abs(unrealized_loss) + total_commission

            # Check max risk constraint
            if max_risk_usd is not None and risk_usd > max_risk_usd:
                print(f"[GridCalculator] Stopping at level {level_num} (max risk ${max_risk_usd:.2f} reached)")
                break

            grid.append(GridLevel(
                level=level_num,
                price=level_price,
                distance_pips=distance_pips,
                lot_size=level_lot,
                total_lot=cumulative_lot,
                avg_price=avg_price,
                breakeven_price=breakeven_price,
                required_reversion_pips=required_reversion_pips,
                reversal_probability=reversal_prob,
                risk_usd=risk_usd
            ))

        return grid

    def print_grid(self, grid: List[GridLevel], title: str = "DCA GRID"):
        """Print grid in formatted table"""
        print("\n" + "="*120)
        print(title)
        print("="*120)

        if not grid:
            print("Empty grid")
            return

        print(f"\n{'Level':<6} {'Price':<10} {'Distance':<10} {'Lot':<8} {'Total Lot':<10} "
              f"{'Avg Price':<10} {'BE Price':<10} {'BE Pips':<8} {'Rev Prob':<8} {'Risk $':<10}")
        print("-" * 120)

        for level in grid:
            print(f"{level.level:<6} {level.price:<10.5f} {level.distance_pips:>8.0f}p {level.lot_size:<8.2f} "
                  f"{level.total_lot:<10.2f} {level.avg_price:<10.5f} {level.breakeven_price:<10.5f} "
                  f"{level.required_reversion_pips:>6.0f}p {level.reversal_probability*100:>6.0f}% "
                  f"${level.risk_usd:>8.2f}")

        print("\n[SUMMARY]")
        final_level = grid[-1]
        print(f"Total Levels: {len(grid)}")
        print(f"Total Lot Size: {final_level.total_lot:.2f}")
        print(f"Average Entry: {final_level.avg_price:.5f}")
        print(f"Breakeven: {final_level.breakeven_price:.5f} ({final_level.required_reversion_pips:.0f} pips from last level)")
        print(f"Max Risk: ${final_level.risk_usd:.2f}")
        print(f"Reversal Probability (at last level): {final_level.reversal_probability*100:.0f}%")
        print("="*120 + "\n")

    def optimize_grid(self,
                     entry_price: float,
                     entry_lot: float,
                     direction: str,
                     current_idx: int,
                     target_recovery_usd: float,
                     max_risk_usd: float) -> List[GridLevel]:
        """
        Optimize grid to achieve target recovery within risk limit

        Args:
            entry_price: Initial entry price
            entry_lot: Initial lot size
            direction: Trade direction
            current_idx: Current index
            target_recovery_usd: Target profit in USD
            max_risk_usd: Maximum risk allowed

        Returns:
            Optimized grid
        """
        print(f"\n[GridCalculator] Optimizing grid for ${target_recovery_usd:.2f} recovery (max risk: ${max_risk_usd:.2f})")

        # Try different configurations
        best_grid = None
        best_score = -999999

        for n_levels in [3, 4, 5, 6]:
            for lot_mult in [1.0, 1.3, 1.5, 2.0]:
                for spacing in ['statistical', 'hybrid']:
                    try:
                        grid = self.calculate_grid(
                            entry_price=entry_price,
                            entry_lot=entry_lot,
                            direction=direction,
                            current_idx=current_idx,
                            n_levels=n_levels,
                            lot_multiplier=lot_mult,
                            max_risk_usd=max_risk_usd,
                            spacing_mode=spacing
                        )

                        if not grid:
                            continue

                        final = grid[-1]

                        # Calculate potential profit at breakeven
                        potential_profit = final.required_reversion_pips * final.total_lot * self.pip_value_in_usd

                        # Score = profit potential - risk, weighted by reversal probability
                        score = (potential_profit - final.risk_usd) * final.reversal_probability

                        # Penalty if doesn't meet target
                        if potential_profit < target_recovery_usd:
                            score *= 0.5

                        if score > best_score:
                            best_score = score
                            best_grid = grid

                    except Exception as e:
                        continue

        if best_grid:
            print(f"[GridCalculator] Optimization complete (score: {best_score:.2f})")
        else:
            print(f"[GridCalculator] Warning: Could not find valid grid within constraints")

        return best_grid


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.append('/home/user/bot_fvg_ruled_base')
    from config import DATA_DIR
    from core.recovery.statistical_analyzer import StatisticalAnalyzer
    from core.recovery.reversal_predictor import ReversalPredictor

    print("\n" + "="*120)
    print("GRID CALCULATOR - EXAMPLE")
    print("="*120)

    # Load data
    data_file = DATA_DIR / 'GBPUSD_M15_180days.csv'

    if not data_file.exists():
        print(f"\nError: Data file not found: {data_file}")
        sys.exit(1)

    print(f"\nLoading data...")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} candles")

    # Initialize components
    print("\nInitializing components...")
    stat_analyzer = StatisticalAnalyzer(data, pip_value=0.0001)
    rev_predictor = ReversalPredictor(mode='rule-based')
    rev_predictor.initialize(data, stat_analyzer)

    calculator = GridCalculator(
        statistical_analyzer=stat_analyzer,
        reversal_predictor=rev_predictor,
        pip_value=0.0001,
        pip_value_in_usd=10.0,
        commission_per_lot=7.0
    )

    # Example 1: SELL trade that went wrong
    print("\n" + "="*120)
    print("EXAMPLE 1: SELL @ 1.2500, price moved up against us")
    print("="*120)

    grid = calculator.calculate_grid(
        entry_price=1.2500,
        entry_lot=0.1,
        direction='sell',
        current_idx=5000,
        n_levels=5,
        lot_multiplier=1.5,
        max_risk_usd=500,
        spacing_mode='statistical'
    )

    calculator.print_grid(grid, "SELL Recovery Grid")

    # Example 2: Optimized grid
    print("\n" + "="*120)
    print("EXAMPLE 2: Optimize grid for $200 recovery, max risk $400")
    print("="*120)

    optimized_grid = calculator.optimize_grid(
        entry_price=1.2500,
        entry_lot=0.1,
        direction='sell',
        current_idx=5000,
        target_recovery_usd=200,
        max_risk_usd=400
    )

    if optimized_grid:
        calculator.print_grid(optimized_grid, "OPTIMIZED Recovery Grid")

    print("\n" + "="*120)
    print("DONE")
    print("="*120 + "\n")
