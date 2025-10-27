"""
DCA Position Manager for Smart Recovery System

Manages multiple DCA positions:
- Tracks entry prices, lot sizes, and average price
- Monitors grid levels and triggers
- Calculates P&L and breakeven
- Handles partial and full position closes

Author: Claude Code
Date: 2025-10-27
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PositionStatus(Enum):
    """Position status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"


@dataclass
class DCAEntry:
    """Single DCA entry"""
    entry_id: int
    timestamp: datetime
    price: float
    lot_size: float
    level: int  # Grid level (0 = initial entry)
    commission: float = 0.0

    def __str__(self):
        return f"Entry #{self.entry_id} L{self.level}: {self.lot_size:.2f} lots @ {self.price:.5f}"


@dataclass
class DCAPosition:
    """
    Multi-entry DCA position

    Tracks:
    - Multiple entries at different prices
    - Average entry price
    - Total lot size
    - Unrealized P&L
    - Breakeven price
    - Grid levels (pending and filled)
    """
    position_id: int
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    initial_entry: DCAEntry
    status: PositionStatus = PositionStatus.OPEN

    # DCA tracking
    entries: List[DCAEntry] = field(default_factory=list)
    grid_levels: List = field(default_factory=list)  # List of GridLevel objects
    next_grid_level: int = 1  # Next grid level to fill

    # Position metrics
    total_lot_size: float = 0.0
    average_entry_price: float = 0.0
    total_commission: float = 0.0

    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Exit tracking
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: str = ""

    def __post_init__(self):
        """Initialize position with initial entry"""
        if not self.entries:
            self.entries = [self.initial_entry]
            self._recalculate()

    def _recalculate(self):
        """Recalculate position metrics"""
        if not self.entries:
            self.total_lot_size = 0.0
            self.average_entry_price = 0.0
            self.total_commission = 0.0
            return

        # Calculate average entry price (weighted)
        total_cost = sum(e.price * e.lot_size for e in self.entries)
        self.total_lot_size = sum(e.lot_size for e in self.entries)
        self.average_entry_price = total_cost / self.total_lot_size if self.total_lot_size > 0 else 0.0

        # Calculate total commission
        self.total_commission = sum(e.commission for e in self.entries)

    def add_entry(self, entry: DCAEntry):
        """Add new DCA entry to position"""
        self.entries.append(entry)
        self._recalculate()

        # Update next grid level
        if entry.level >= self.next_grid_level:
            self.next_grid_level = entry.level + 1

    def update_pnl(self, current_price: float, pip_value: float, pip_value_in_usd: float):
        """
        Update unrealized P&L

        Args:
            current_price: Current market price
            pip_value: Pip value (0.0001)
            pip_value_in_usd: USD per pip per lot
        """
        if self.status == PositionStatus.CLOSED:
            # Use exit price for closed positions
            price_diff = self.exit_price - self.average_entry_price if self.direction == 'BUY' else self.average_entry_price - self.exit_price
        else:
            # Use current price for open positions
            price_diff = current_price - self.average_entry_price if self.direction == 'BUY' else self.average_entry_price - current_price

        pips = price_diff / pip_value
        gross_pnl = pips * self.total_lot_size * pip_value_in_usd
        self.unrealized_pnl = gross_pnl - self.total_commission

    def get_breakeven_price(self, pip_value: float, pip_value_in_usd: float) -> float:
        """
        Calculate breakeven price (including commission)

        Args:
            pip_value: Pip value
            pip_value_in_usd: USD per pip per lot

        Returns:
            Breakeven price
        """
        if self.total_lot_size == 0:
            return self.average_entry_price

        # Calculate pips needed to cover commission
        commission_pips = self.total_commission / (self.total_lot_size * pip_value_in_usd)

        # Adjust average entry by commission
        if self.direction == 'BUY':
            return self.average_entry_price + (commission_pips * pip_value)
        else:
            return self.average_entry_price - (commission_pips * pip_value)

    def get_next_grid_price(self) -> Optional[float]:
        """Get price of next grid level to fill"""
        if self.next_grid_level <= len(self.grid_levels):
            return self.grid_levels[self.next_grid_level - 1].price
        return None

    def should_add_dca_entry(self, current_price: float) -> bool:
        """
        Check if should add DCA entry at current price

        Args:
            current_price: Current market price

        Returns:
            True if should add entry
        """
        if self.status != PositionStatus.OPEN:
            return False

        next_price = self.get_next_grid_price()
        if next_price is None:
            return False  # No more grid levels

        # Check if price has reached next grid level
        if self.direction == 'BUY':
            return current_price <= next_price
        else:
            return current_price >= next_price

    def should_close(self,
                    current_price: float,
                    pip_value: float,
                    pip_value_in_usd: float,
                    min_profit_pips: float = 10) -> bool:
        """
        Check if should close position (price has reverted to profitable zone)

        Args:
            current_price: Current market price
            pip_value: Pip value
            pip_value_in_usd: USD per pip per lot
            min_profit_pips: Minimum profit in pips to close

        Returns:
            True if should close
        """
        if self.status != PositionStatus.OPEN:
            return False

        # Calculate breakeven
        breakeven = self.get_breakeven_price(pip_value, pip_value_in_usd)

        # Check if past breakeven + min profit
        if self.direction == 'BUY':
            target_price = breakeven + (min_profit_pips * pip_value)
            return current_price >= target_price
        else:
            target_price = breakeven - (min_profit_pips * pip_value)
            return current_price <= target_price

    def close_position(self,
                      exit_price: float,
                      exit_timestamp: datetime,
                      exit_reason: str,
                      pip_value: float,
                      pip_value_in_usd: float):
        """
        Close entire position

        Args:
            exit_price: Exit price
            exit_timestamp: Exit timestamp
            exit_reason: Reason for exit
            pip_value: Pip value
            pip_value_in_usd: USD per pip per lot
        """
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.exit_reason = exit_reason
        self.status = PositionStatus.CLOSED

        # Calculate final P&L
        self.update_pnl(exit_price, pip_value, pip_value_in_usd)
        self.realized_pnl = self.unrealized_pnl
        self.unrealized_pnl = 0.0

    def __str__(self):
        entries_str = ", ".join([f"L{e.level}:{e.lot_size:.2f}@{e.price:.5f}" for e in self.entries])
        return (f"Position #{self.position_id} {self.direction} {self.status.value}: "
                f"[{entries_str}] Avg: {self.average_entry_price:.5f}, "
                f"Total: {self.total_lot_size:.2f} lots, PnL: ${self.unrealized_pnl + self.realized_pnl:.2f}")


class DCAPositionManager:
    """
    Manage multiple DCA positions

    Features:
    - Open new DCA positions with grid
    - Add DCA entries when price hits grid levels
    - Monitor reversal signals and close positions
    - Track performance metrics

    Usage:
        manager = DCAPositionManager(
            grid_calculator=grid_calc,
            reversal_predictor=rev_pred,
            pip_value=0.0001,
            pip_value_in_usd=10.0,
            commission_per_lot=7.0
        )

        # Open position
        position = manager.open_position(
            symbol='GBPUSD',
            direction='SELL',
            entry_price=1.2500,
            entry_lot=0.1,
            entry_timestamp=datetime.now(),
            current_idx=1000,
            n_grid_levels=5,
            lot_multiplier=1.5
        )

        # Update on each bar
        manager.update(current_price=1.2550, current_timestamp=datetime.now())

        # Check if positions should close
        if manager.should_close_any():
            closed = manager.close_profitable_positions()
    """

    def __init__(self,
                 grid_calculator,
                 reversal_predictor,
                 pip_value: float = 0.0001,
                 pip_value_in_usd: float = 10.0,
                 commission_per_lot: float = 7.0):
        """
        Initialize position manager

        Args:
            grid_calculator: GridCalculator instance
            reversal_predictor: ReversalPredictor instance
            pip_value: Pip value
            pip_value_in_usd: USD per pip per lot
            commission_per_lot: Commission per lot
        """
        self.grid_calculator = grid_calculator
        self.reversal_predictor = reversal_predictor
        self.pip_value = pip_value
        self.pip_value_in_usd = pip_value_in_usd
        self.commission_per_lot = commission_per_lot

        # Positions tracking
        self.positions: List[DCAPosition] = []
        self.next_position_id = 1
        self.next_entry_id = 1

        # Performance tracking
        self.total_positions_opened = 0
        self.total_positions_closed = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

    def open_position(self,
                     symbol: str,
                     direction: str,
                     entry_price: float,
                     entry_lot: float,
                     entry_timestamp: datetime,
                     current_idx: int,
                     n_grid_levels: int = 5,
                     lot_multiplier: float = 1.5,
                     max_risk_usd: Optional[float] = None) -> DCAPosition:
        """
        Open new DCA position with grid

        Args:
            symbol: Symbol name
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            entry_lot: Entry lot size
            entry_timestamp: Entry timestamp
            current_idx: Current index in data
            n_grid_levels: Number of grid levels
            lot_multiplier: Lot size multiplier
            max_risk_usd: Maximum risk

        Returns:
            New DCAPosition
        """
        # Calculate grid
        grid = self.grid_calculator.calculate_grid(
            entry_price=entry_price,
            entry_lot=entry_lot,
            direction=direction,
            current_idx=current_idx,
            n_levels=n_grid_levels,
            lot_multiplier=lot_multiplier,
            max_risk_usd=max_risk_usd
        )

        # Create initial entry
        initial_entry = DCAEntry(
            entry_id=self.next_entry_id,
            timestamp=entry_timestamp,
            price=entry_price,
            lot_size=entry_lot,
            level=0,  # Initial entry
            commission=entry_lot * self.commission_per_lot
        )
        self.next_entry_id += 1

        # Create position
        position = DCAPosition(
            position_id=self.next_position_id,
            symbol=symbol,
            direction=direction,
            initial_entry=initial_entry,
            grid_levels=grid
        )
        self.next_position_id += 1

        self.positions.append(position)
        self.total_positions_opened += 1

        return position

    def update(self, current_price: float, current_timestamp: datetime, current_idx: int):
        """
        Update all open positions

        Args:
            current_price: Current market price
            current_timestamp: Current timestamp
            current_idx: Current index in data
        """
        for position in self.positions:
            if position.status != PositionStatus.OPEN:
                continue

            # Update P&L
            position.update_pnl(current_price, self.pip_value, self.pip_value_in_usd)

            # Check if should add DCA entry
            if position.should_add_dca_entry(current_price):
                self._add_dca_entry(position, current_price, current_timestamp, current_idx)

    def _add_dca_entry(self, position: DCAPosition, price: float, timestamp: datetime, current_idx: int):
        """Add DCA entry to position"""
        next_level = position.next_grid_level

        if next_level <= len(position.grid_levels):
            grid_level = position.grid_levels[next_level - 1]

            # Create entry
            entry = DCAEntry(
                entry_id=self.next_entry_id,
                timestamp=timestamp,
                price=price,
                lot_size=grid_level.lot_size,
                level=next_level,
                commission=grid_level.lot_size * self.commission_per_lot
            )
            self.next_entry_id += 1

            position.add_entry(entry)

            print(f"[DCA] Added entry to Position #{position.position_id}: {entry}")
            print(f"      New average: {position.average_entry_price:.5f}, Total lot: {position.total_lot_size:.2f}")

    def should_close_any(self, current_price: float, min_profit_pips: float = 10) -> bool:
        """Check if any position should close"""
        for position in self.positions:
            if position.should_close(current_price, self.pip_value, self.pip_value_in_usd, min_profit_pips):
                return True
        return False

    def close_profitable_positions(self,
                                   current_price: float,
                                   current_timestamp: datetime,
                                   min_profit_pips: float = 10) -> List[DCAPosition]:
        """
        Close all positions that have reached profit target

        Args:
            current_price: Current market price
            current_timestamp: Current timestamp
            min_profit_pips: Minimum profit in pips

        Returns:
            List of closed positions
        """
        closed = []

        for position in self.positions:
            if position.should_close(current_price, self.pip_value, self.pip_value_in_usd, min_profit_pips):
                position.close_position(
                    exit_price=current_price,
                    exit_timestamp=current_timestamp,
                    exit_reason=f"Profit target ({min_profit_pips}p)",
                    pip_value=self.pip_value,
                    pip_value_in_usd=self.pip_value_in_usd
                )
                closed.append(position)

                self.total_positions_closed += 1
                if position.realized_pnl > 0:
                    self.total_profit += position.realized_pnl
                else:
                    self.total_loss += abs(position.realized_pnl)

                print(f"\n[CLOSE] Position #{position.position_id} closed:")
                print(f"        Exit: {current_price:.5f} ({position.exit_reason})")
                print(f"        PnL: ${position.realized_pnl:.2f}")
                print(f"        Entries: {len(position.entries)}")

        return closed

    def get_open_positions(self) -> List[DCAPosition]:
        """Get all open positions"""
        return [p for p in self.positions if p.status == PositionStatus.OPEN]

    def get_closed_positions(self) -> List[DCAPosition]:
        """Get all closed positions"""
        return [p for p in self.positions if p.status == PositionStatus.CLOSED]

    def print_summary(self):
        """Print performance summary"""
        open_pos = self.get_open_positions()
        closed_pos = self.get_closed_positions()

        print("\n" + "="*100)
        print("DCA POSITION MANAGER - SUMMARY")
        print("="*100)

        print(f"\n[POSITIONS]")
        print(f"  Total Opened: {self.total_positions_opened}")
        print(f"  Currently Open: {len(open_pos)}")
        print(f"  Total Closed: {self.total_positions_closed}")

        print(f"\n[PERFORMANCE]")
        if self.total_positions_closed > 0:
            net_pnl = self.total_profit - self.total_loss
            win_rate = (self.total_profit / (self.total_profit + self.total_loss) * 100
                       if (self.total_profit + self.total_loss) > 0 else 0)

            print(f"  Total Profit: ${self.total_profit:.2f}")
            print(f"  Total Loss: ${self.total_loss:.2f}")
            print(f"  Net P&L: ${net_pnl:.2f}")
            print(f"  Win Rate: {win_rate:.1f}%")
        else:
            print(f"  No closed positions yet")

        if open_pos:
            print(f"\n[OPEN POSITIONS]")
            for position in open_pos:
                print(f"  {position}")

        print("="*100 + "\n")


# Example usage
if __name__ == '__main__':
    print("\n" + "="*100)
    print("DCA POSITION MANAGER - EXAMPLE")
    print("="*100)

    print("\n[INFO] This module requires initialized GridCalculator and ReversalPredictor")
    print("[INFO] See examples/run_dca_backtest.py for complete usage example")
    print("\n" + "="*100 + "\n")
