"""
Backtest Engine with Virtual/Real Mode and Martingale

Engine backtest hoan chinh voi:
- Virtual mode: test voi lot size that
- Real mode: auto switch khi thua 3 lenh lien tiep, martingale x1.3
- Position sizing based on ATR
- SL/TP management
- Trade logging
- Performance metrics

Author: Claude Code
Date: 2025-10-25
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TradeMode(Enum):
    """Trade mode"""
    VIRTUAL = "VIRTUAL"  # Test mode with real lot size
    REAL = "REAL"        # After 3 consecutive losses


@dataclass
class Trade:
    """Trade record with full config parameters for AI analysis"""
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

    # Config parameters (for AI analysis & optimization)
    config_params: dict = field(default_factory=dict)

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
        """Convert to dictionary with all config parameters"""
        base_dict = {
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

        # Add config parameters for AI analysis
        base_dict.update(self.config_params)

        return base_dict


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_balance: float = 1000.0      # Starting balance
    risk_per_trade: float = 0.02         # 2% risk per trade
    base_lot_size: float = 0.1           # Base lot size in virtual mode

    # Dynamic Risk Recovery settings
    consecutive_losses_trigger: int = 3  # Switch to real mode after N losses
    recovery_multiplier: float = 2.0     # Recovery target = Total Loss × 2.0
    min_lot_size: float = 0.01           # Minimum lot size (broker limit)
    max_lot_size: float = 10.0           # Maximum lot size limit

    # Trading settings
    pip_value: float = 0.0001            # For 5-digit broker (price increment)
    pip_value_in_account_currency: float = 10.0  # $ per pip per standard lot
    commission_per_lot: float = 7.0      # Commission per lot (round trip)

    # Stop loss / Take profit mode
    use_atr_sl_tp: bool = False          # True=ATR mode, False=Pips mode

    # ATR mode settings
    atr_sl_multiplier: float = 1.5       # SL = ATR * 1.5
    atr_tp_multiplier: float = 3.0       # TP = ATR * 3.0

    # Pips mode settings
    sl_pips: float = 50                  # SL in pips (fixed mode)
    tp_pips: float = 100                 # TP in pips (fixed mode)

    # Confluence scoring thresholds
    min_confidence_score: float = 70.0   # Minimum score to trade


class Backtester:
    """
    Backtest engine with Dynamic Risk Recovery System

    Virtual Mode (Default):
    - Trade with base_lot_size
    - Track performance without risking real capital
    - Accumulate virtual losses

    Real Mode (After N consecutive virtual losses):
    - Calculate lot size dynamically to recover all losses + profit
    - Formula: lot_size = (total_loss × recovery_multiplier) / (tp_pips × pip_value)
    - Minimum $10 profit guarantee
    - After win: reset to virtual mode and clear all losses
    - Lot sizes rounded to 2 decimal places (e.g., 0.013 → 0.01)

    SL/TP Modes:
    - ATR Mode: Dynamic SL/TP based on market volatility
    - Pips Mode: Fixed SL/TP in pips
    """

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtester

        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()

        # Account state
        self.balance = self.config['initial_balance']
        self.equity = self.config['initial_balance']
        self.peak_balance = self.config['initial_balance']

        # Trading state
        self.mode = TradeMode.VIRTUAL
        self.consecutive_losses = 0
        self.current_lot_size = self.config['base_lot_size']

        # Dynamic Risk Recovery
        self.total_virtual_loss = 0.0  # Track total loss in VIRTUAL mode
        self.total_real_loss = 0.0     # Track total loss in REAL mode (for recovery calculation)

        # Trade tracking
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None  # Renamed to avoid conflict with open_trade() method

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0

    def round_lot_size(self, lot_size: float) -> float:
        """
        Round lot size to 2 decimal places
        Examples: 0.013 → 0.01, 0.017 → 0.02, 0.155 → 0.16

        Args:
            lot_size: Raw lot size

        Returns:
            Rounded lot size (2 decimals)
        """
        import math
        # Round to 2 decimal places (0.01 precision)
        rounded = round(lot_size, 2)

        # Ensure minimum lot size
        if rounded < self.config['min_lot_size']:
            rounded = self.config['min_lot_size']

        # Ensure maximum lot size
        if rounded > self.config['max_lot_size']:
            rounded = self.config['max_lot_size']

        return rounded

    def calculate_sl_tp(self, current_price: float, direction: str, atr_value: float = None):
        """
        Calculate SL and TP based on mode (ATR or PIPS)

        Args:
            current_price: Current market price
            direction: 'BUY' or 'SELL'
            atr_value: ATR value (required if use_atr_sl_tp = True)

        Returns:
            tuple: (sl_price, tp_price, sl_distance_pips, tp_distance_pips)
        """
        pip_value = self.config['pip_value']

        if self.config['use_atr_sl_tp']:
            # ATR Mode
            if atr_value is None:
                raise ValueError("ATR value required when use_atr_sl_tp = True")

            sl_distance = atr_value * self.config['atr_sl_multiplier']
            tp_distance = atr_value * self.config['atr_tp_multiplier']
        else:
            # PIPS Mode
            sl_distance = self.config['sl_pips'] * pip_value
            tp_distance = self.config['tp_pips'] * pip_value

        # Calculate prices
        if direction == 'BUY':
            sl_price = current_price - sl_distance
            tp_price = current_price + tp_distance
        else:  # SELL
            sl_price = current_price + sl_distance
            tp_price = current_price - tp_distance

        # Calculate distances in pips (for lot size calculation)
        sl_distance_pips = sl_distance / pip_value
        tp_distance_pips = tp_distance / pip_value

        return sl_price, tp_price, sl_distance_pips, tp_distance_pips

    def calculate_recovery_lot_size(self, tp_distance_pips: float) -> float:
        """
        Calculate lot size needed to recover total losses + profit

        Formula:
            Required Profit = (Total Virtual Loss + Total Real Loss) × recovery_multiplier
            Lot Size = Required Profit / (TP Distance in pips × pip_value)

        Args:
            tp_distance_pips: TP distance in pips

        Returns:
            Calculated lot size (rounded to 2 decimals)
        """
        total_loss = abs(self.total_virtual_loss) + abs(self.total_real_loss)

        # Target profit = loss × recovery_multiplier (default 2.0)
        # Example: -$5 loss → need $10 profit (recover $5 + gain $5)
        required_profit = total_loss * self.config['recovery_multiplier']

        # Minimum required profit
        min_required_profit = 10.0  # At least $10 profit as user specified
        if required_profit < min_required_profit:
            required_profit = min_required_profit

        # Calculate lot size
        # Profit per standard lot = TP pips × pip_value_in_account_currency
        # Example: 150 pips × $10/pip = $1,500 profit per 1.0 lot
        # Lot size = required_profit / (TP pips × pip_value_in_account_currency)

        pip_value_in_money = self.config.get('pip_value_in_account_currency', 10.0)
        profit_per_standard_lot = tp_distance_pips * pip_value_in_money

        if profit_per_standard_lot == 0:
            return self.config['base_lot_size']

        raw_lot_size = required_profit / profit_per_standard_lot

        # Round to 2 decimals
        lot_size = self.round_lot_size(raw_lot_size)

        return lot_size

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
        atr_value: float = None
    ) -> Optional[Trade]:
        """
        Open a new trade with dynamic risk recovery

        Args:
            timestamp: Current timestamp
            signal_data: Signal data from ConfluenceScorer
            current_price: Current market price
            atr_value: Current ATR value (required if use_atr_sl_tp = True)

        Returns:
            Trade object if opened, None otherwise
        """
        if not self.should_trade(signal_data):
            return None

        direction = signal_data['signal']

        # Calculate SL and TP (supports both ATR and PIPS modes)
        sl_price, tp_price, sl_distance_pips, tp_distance_pips = self.calculate_sl_tp(
            current_price, direction, atr_value
        )

        # Calculate lot size based on mode
        if self.mode == TradeMode.VIRTUAL:
            # Virtual mode: use base lot size
            lot_size = self.config['base_lot_size']
        else:
            # REAL mode: calculate lot size for recovery
            lot_size = self.calculate_recovery_lot_size(tp_distance_pips)

        # Round lot size to 2 decimals
        lot_size = self.round_lot_size(lot_size)

        # Extract config parameters for CSV logging (AI analysis)
        confluence_weights = self.config.get('confluence_weights', {})
        config_params = {
            'timeframe': self.config.get('timeframe', ''),
            'fvg_timeframe': self.config.get('fvg_timeframe', ''),
            'base_lot_size': self.config.get('base_lot_size', 0),
            'consecutive_losses_trigger': self.config.get('consecutive_losses_trigger', 0),
            'recovery_multiplier': self.config.get('recovery_multiplier', 0),
            'use_atr_sl_tp': self.config.get('use_atr_sl_tp', False),
            'atr_sl_multiplier': self.config.get('atr_sl_multiplier', 0),
            'atr_tp_multiplier': self.config.get('atr_tp_multiplier', 0),
            'sl_pips': self.config.get('sl_pips', 0),
            'tp_pips': self.config.get('tp_pips', 0),
            'min_confidence_score': self.config.get('min_confidence_score', 0),
            'adx_threshold': self.config.get('adx_threshold', 0),
            'use_statistical': self.config.get('use_statistical', False),
            'weight_fvg': confluence_weights.get('fvg', 0),
            # Statistical mode weights
            'weight_fvg_size_atr': confluence_weights.get('fvg_size_atr', 0),
            'weight_hurst': confluence_weights.get('hurst', 0),
            'weight_lr_deviation': confluence_weights.get('lr_deviation', 0),
            'weight_skewness': confluence_weights.get('skewness', 0),
            'weight_kurtosis': confluence_weights.get('kurtosis', 0),
            'weight_obv_div': confluence_weights.get('obv_div', 0),
            'weight_overlap_count': confluence_weights.get('overlap_count', 0),
            'weight_regime': confluence_weights.get('regime', 0),
            # Basic mode weights
            'weight_vwap': confluence_weights.get('vwap', 0),
            'weight_obv': confluence_weights.get('obv', 0),
            'weight_volume': confluence_weights.get('volume', 0),
        }

        # Add raw indicator values (for feature engineering)
        raw_indicators = signal_data.get('raw_indicators', {})
        config_params.update({
            'hurst': raw_indicators.get('hurst', 0),
            'lr_deviation': raw_indicators.get('lr_deviation', 0),
            'r2': raw_indicators.get('r2', 0),
            'skewness': raw_indicators.get('skewness', 0),
            'kurtosis': raw_indicators.get('kurtosis', 0),
            'obv_divergence': raw_indicators.get('obv_divergence', 0),
            'atr_percentile': raw_indicators.get('atr_percentile', 0),
        })

        # Add component scores (for feature engineering)
        components = signal_data.get('components', {})
        config_params.update({
            'score_fvg': components.get('fvg', 0),
            'score_fvg_size_atr': components.get('fvg_size_atr', 0),
            'score_hurst': components.get('hurst', 0),
            'score_lr_deviation': components.get('lr_deviation', 0),
            'score_skewness': components.get('skewness', 0),
            'score_kurtosis': components.get('kurtosis', 0),
            'score_obv_div': components.get('obv_div', 0),
            'score_overlap_count': components.get('overlap_count', 0),
            'score_regime': components.get('regime', 0),
            # Basic mode component scores
            'score_vwap': components.get('vwap', 0),
            'score_obv': components.get('obv', 0),
            'score_volume': components.get('volume', 0),
        })

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
            config_params=config_params,
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

        # Update win/loss counts and dynamic risk recovery
        if trade.is_win():
            self.winning_trades += 1
            self.consecutive_losses = 0

            # Reset to virtual mode and clear losses
            self.mode = TradeMode.VIRTUAL
            self.total_virtual_loss = 0.0
            self.total_real_loss = 0.0
            self.current_lot_size = self.config['base_lot_size']
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1

            # Track losses by mode
            if trade.mode == TradeMode.VIRTUAL:
                # Accumulate virtual losses
                self.total_virtual_loss += trade.pnl  # pnl is negative for losses
            else:
                # Accumulate real losses
                self.total_real_loss += trade.pnl

            # Switch to REAL mode after consecutive losses
            if self.consecutive_losses >= self.config['consecutive_losses_trigger']:
                self.mode = TradeMode.REAL
                # Note: lot size will be calculated dynamically in next open_trade()
                # based on total_virtual_loss + total_real_loss

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
