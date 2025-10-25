"""
Backtest Engine with Virtual/Real Mode and Martingale

Engine backtest hoan chinh voi:
- Virtual mode: test voi lot size that
- Real mode: auto switch khi thua 3 lenh lien tiep, martingale x1.3
- Position sizing based on ATR
- SL/TP management
- Trade logging
- Performance metrics

IMPORTANT: All config now centralized in config.py!
Just import BACKTEST_CONFIG and use it.

Author: Claude Code
Date: 2025-10-25
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import centralized config
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from config import BACKTEST_CONFIG


class TradeMode(Enum):
    """Trade mode"""
    VIRTUAL = "VIRTUAL"  # Test mode with real lot size
    REAL = "REAL"        # After 3 consecutive losses


@dataclass
class Trade:
    """Trade record"""
    entry_time: datetime
    entry_price: float
    direction: str  # 'BUY' or 'SELL'
    lot_size: float
    sl_price: float
    tp_price: float
    mode: TradeMode

    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'END'
    pnl: Optional[float] = None
    pnl_pips: Optional[float] = None

    # Signal analysis data
    confluence_score: float = 0.0
    confidence: str = ""
    fvg_bias: str = ""

    def close(self, exit_time: datetime, exit_price: float, exit_reason: str, pip_value: float = 0.0001):
        """Close the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate PnL
        if self.direction == 'BUY':
            self.pnl_pips = (exit_price - self.entry_price) / pip_value
        else:  # SELL
            self.pnl_pips = (self.entry_price - exit_price) / pip_value

        # PnL in currency (simplified: $10 per pip per lot)
        self.pnl = self.pnl_pips * 10 * self.lot_size

    def is_win(self) -> bool:
        """Check if trade is winning"""
        return self.pnl > 0 if self.pnl is not None else False

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'lot_size': self.lot_size,
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'mode': self.mode.value,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pips': self.pnl_pips,
            'confluence_score': self.confluence_score,
            'confidence': self.confidence,
            'fvg_bias': self.fvg_bias,
        }


class Backtester:
    """
    Backtest engine with virtual/real mode switching

    Virtual Mode (Default):
    - Trade with base_lot_size
    - Track performance

    Real Mode (After 3 consecutive losses):
    - Start with base_lot_size
    - After each loss: lot_size * martingale_multiplier
    - After win: reset to base_lot_size and back to virtual mode
    """

    def __init__(self, config: dict = None):
        """
        Initialize backtester

        Args:
            config: Backtest configuration dict (uses BACKTEST_CONFIG from config.py if None)
        """
        # Use centralized config from config.py
        self.config = config or BACKTEST_CONFIG

        # Account state
        self.balance = self.config['initial_balance']
        self.equity = self.config['initial_balance']
        self.peak_balance = self.config['initial_balance']

        # Trading state
        self.mode = TradeMode.VIRTUAL
        self.consecutive_losses = 0
        self.current_lot_size = self.config['base_lot_size']

        # Trade tracking
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None  # Renamed to avoid conflict with open_trade() method

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0

    def should_trade(self, signal_data: dict) -> bool:
        """
        Check if should open a trade based on signal

        Args:
            signal_data: Signal data from strategy

        Returns:
            bool: True if should trade
        """
        # Already have open trade
        if self.current_trade is not None:
            return False

        # Check signal exists
        if signal_data.get('signal') == 'NEUTRAL':
            return False

        # Check confidence score
        score = signal_data.get('total_score', 0)
        if score < self.config['min_confidence_score']:
            return False

        return True

    def open_trade(
        self,
        timestamp: datetime,
        signal_data: dict,
        current_price: float,
        atr_value: float
    ) -> Optional[Trade]:
        """
        Open a new trade

        Args:
            timestamp: Current timestamp
            signal_data: Signal data from ConfluenceScorer
            current_price: Current market price
            atr_value: Current ATR value

        Returns:
            Trade object if opened, None otherwise
        """
        if not self.should_trade(signal_data):
            return None

        direction = signal_data['signal']

        # Calculate SL and TP based on ATR
        sl_distance = atr_value * self.config['atr_sl_multiplier']
        tp_distance = atr_value * self.config['atr_tp_multiplier']

        if direction == 'BUY':
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:  # SELL
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance

        # Calculate lot size based on mode
        if self.mode == TradeMode.VIRTUAL:
            lot_size = self.config['base_lot_size']
        else:  # REAL mode with martingale
            lot_size = min(self.current_lot_size, self.config['max_lot_size'])

        # Create trade
        trade = Trade(
            entry_time=timestamp,
            entry_price=current_price,
            direction=direction,
            lot_size=lot_size,
            sl_price=sl_price,
            tp_price=tp_price,
            mode=self.mode,
            confluence_score=signal_data.get('total_score', 0),
            confidence=signal_data.get('confidence', ''),
            fvg_bias=signal_data.get('fvg_structure', {}).get('bias', ''),
        )

        self.current_trade = trade
        self.total_trades += 1

        return trade

    def update_open_trade(
        self,
        timestamp: datetime,
        high: float,
        low: float,
        close: float
    ) -> Optional[Trade]:
        """
        Update open trade - check for SL/TP hit

        Args:
            timestamp: Current timestamp
            high: Current candle high
            low: Current candle low
            close: Current candle close

        Returns:
            Closed trade if closed, None otherwise
        """
        if self.current_trade is None:
            return None

        trade = self.current_trade
        closed = False
        exit_price = None
        exit_reason = None

        if trade.direction == 'BUY':
            # Check SL hit
            if low <= trade.sl_price:
                closed = True
                exit_price = trade.sl_price
                exit_reason = 'SL'
            # Check TP hit
            elif high >= trade.tp_price:
                closed = True
                exit_price = trade.tp_price
                exit_reason = 'TP'

        else:  # SELL
            # Check SL hit
            if high >= trade.sl_price:
                closed = True
                exit_price = trade.sl_price
                exit_reason = 'SL'
            # Check TP hit
            elif low <= trade.tp_price:
                closed = True
                exit_price = trade.tp_price
                exit_reason = 'TP'

        if closed:
            return self.close_trade(timestamp, exit_price, exit_reason)

        return None

    def close_trade(
        self,
        timestamp: datetime,
        exit_price: float,
        exit_reason: str
    ) -> Trade:
        """
        Close the open trade

        Args:
            timestamp: Exit timestamp
            exit_price: Exit price
            exit_reason: Exit reason ('TP', 'SL', 'END')

        Returns:
            Closed trade
        """
        if self.current_trade is None:
            return None

        trade = self.current_trade
        trade.close(timestamp, exit_price, exit_reason, self.config['pip_value'])

        # Apply commission
        commission = self.config['commission_per_lot'] * trade.lot_size
        trade.pnl -= commission

        # Update balance
        self.balance += trade.pnl
        self.equity = self.balance
        self.total_pnl += trade.pnl

        # Update peak and drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Update win/loss counts
        if trade.is_win():
            self.winning_trades += 1
            self.consecutive_losses = 0

            # Reset to virtual mode after win in real mode
            if self.mode == TradeMode.REAL:
                self.mode = TradeMode.VIRTUAL
                self.current_lot_size = self.config['base_lot_size']
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1

            # Switch to real mode after consecutive losses
            if self.consecutive_losses >= self.config['consecutive_losses_trigger']:
                if self.mode == TradeMode.VIRTUAL:
                    # First loss in real mode - start with base size
                    self.mode = TradeMode.REAL
                    self.current_lot_size = self.config['base_lot_size']
                else:
                    # Subsequent losses in real mode - apply martingale
                    self.current_lot_size *= self.config['martingale_multiplier']

        # Add to trades history
        self.trades.append(trade)
        self.current_trade = None

        return trade

    def get_performance_metrics(self) -> dict:
        """
        Calculate performance metrics

        Returns:
            dict: Performance metrics
        """
        if self.total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'final_balance': self.balance,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
            }

        # Calculate wins and losses
        wins = [t.pnl for t in self.trades if t.is_win()]
        losses = [t.pnl for t in self.trades if not t.is_win()]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,

            'initial_balance': self.config['initial_balance'],
            'final_balance': self.balance,
            'total_pnl': self.total_pnl,
            'return_pct': ((self.balance - self.config['initial_balance']) / self.config['initial_balance'] * 100),

            'max_drawdown': self.max_drawdown * 100,
            'profit_factor': profit_factor,

            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'largest_win': max(wins) if wins else 0,
            'largest_loss': min(losses) if losses else 0,

            'avg_win_pips': np.mean([t.pnl_pips for t in self.trades if t.is_win()]) if wins else 0,
            'avg_loss_pips': np.mean([t.pnl_pips for t in self.trades if not t.is_win()]) if losses else 0,
        }

    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get trades as DataFrame

        Returns:
            pd.DataFrame: Trades data
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self.trades])

    def print_summary(self):
        """Print backtest summary"""
        metrics = self.get_performance_metrics()

        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)

        print(f"\n[TRADES] Trading Statistics:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Winning Trades: {metrics['winning_trades']}")
        print(f"   Losing Trades: {metrics['losing_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.2f}%")

        print(f"\n[PROFIT] Financial Results:")
        print(f"   Initial Balance: ${metrics['initial_balance']:,.2f}")
        print(f"   Final Balance: ${metrics['final_balance']:,.2f}")
        print(f"   Total PnL: ${metrics['total_pnl']:,.2f}")
        print(f"   Return: {metrics['return_pct']:.2f}%")

        print(f"\n[METRICS] Performance Metrics:")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Avg Win: ${metrics['avg_win']:.2f} ({metrics['avg_win_pips']:.1f} pips)")
        print(f"   Avg Loss: ${metrics['avg_loss']:.2f} ({metrics['avg_loss_pips']:.1f} pips)")
        print(f"   Largest Win: ${metrics['largest_win']:.2f}")
        print(f"   Largest Loss: ${metrics['largest_loss']:.2f}")

        print("="*80)
