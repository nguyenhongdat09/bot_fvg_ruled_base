"""
Backtester - Main backtesting engine

Supports:
- Fixed SL/TP in pips
- ATR-based SL/TP
- Trade management
- Performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Trade object to track individual trades"""
    trade_id: int
    signal: Literal['BUY', 'SELL']
    entry_time: pd.Timestamp
    entry_price: float
    entry_index: int
    lot_size: float
    
    # SL/TP levels
    sl_price: float
    tp_price: float
    
    # Exit info
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_index: Optional[int] = None
    exit_reason: Optional[str] = None  # 'TP', 'SL', 'CLOSE'
    
    # P&L
    pnl: float = 0.0
    pnl_pips: float = 0.0
    commission: float = 0.0
    
    # Status
    is_open: bool = True
    
    def calculate_pnl(self, exit_price: float, pip_value: float, lot_size_value: float):
        """Calculate P&L for the trade"""
        if self.signal == 'BUY':
            pnl_pips = (exit_price - self.entry_price) / pip_value
        else:  # SELL
            pnl_pips = (self.entry_price - exit_price) / pip_value
        
        self.pnl_pips = pnl_pips
        self.pnl = pnl_pips * pip_value * lot_size_value * self.lot_size - self.commission
        return self.pnl


class Backtester:
    """
    Main backtesting engine
    
    Features:
    - Fixed or ATR-based SL/TP
    - Risk-based position sizing
    - Trade tracking
    - Performance metrics
    """
    
    def __init__(self, config: Dict):
        """
        Initialize backtester
        
        Args:
            config: BACKTEST_CONFIG dictionary
        """
        self.config = config
        
        # Capital management
        self.initial_capital = config['initial_capital']
        self.current_capital = config['initial_capital']
        self.equity_curve = []
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.trade_counter = 0
        
        # Symbol info
        self.pip_value = config['pip_value']
        self.lot_size = config['lot_size']
        self.spread_pips = config['spread_pips']
        
        # SL/TP mode
        self.sl_tp_mode = config['sl_tp_mode']
        
    def calculate_lot_size(self, sl_pips: float) -> float:
        """
        Calculate position size based on risk
        
        Args:
            sl_pips: Stop loss in pips
            
        Returns:
            Lot size
        """
        risk_amount = self.current_capital * (self.config['risk_per_trade_percent'] / 100.0)
        
        # Risk per pip = pip_value × lot_size
        # Total risk = risk_per_pip × sl_pips
        # lot_size = risk_amount / (pip_value × lot_size × sl_pips)
        lot = risk_amount / (self.pip_value * self.lot_size * sl_pips)
        
        # Clamp to min/max
        lot = max(self.config['min_lot'], min(lot, self.config['max_lot']))
        
        return round(lot, 2)
    
    def calculate_sl_tp(self, signal: str, entry_price: float, 
                       atr_value: Optional[float] = None) -> tuple:
        """
        Calculate SL and TP prices based on mode
        
        Args:
            signal: 'BUY' or 'SELL'
            entry_price: Entry price
            atr_value: ATR value (required if mode='atr')
            
        Returns:
            (sl_price, tp_price, sl_pips, tp_pips)
        """
        if self.sl_tp_mode == 'fixed':
            # Fixed pips mode
            sl_pips = self.config['fixed_sl_pips']
            tp_pips = self.config['fixed_tp_pips']
        else:  # atr mode
            if atr_value is None:
                raise ValueError("ATR value required for ATR-based SL/TP")
            
            # Convert ATR to pips
            atr_pips = atr_value / self.pip_value
            sl_pips = atr_pips * self.config['atr_sl_multiplier']
            tp_pips = atr_pips * self.config['atr_tp_multiplier']
        
        # Calculate actual prices
        if signal == 'BUY':
            sl_price = entry_price - (sl_pips * self.pip_value)
            tp_price = entry_price + (tp_pips * self.pip_value)
        else:  # SELL
            sl_price = entry_price + (sl_pips * self.pip_value)
            tp_price = entry_price - (tp_pips * self.pip_value)
        
        return sl_price, tp_price, sl_pips, tp_pips
    
    def open_trade(self, signal: str, entry_price: float, entry_time: pd.Timestamp,
                   entry_index: int, atr_value: Optional[float] = None) -> Optional[Trade]:
        """
        Open a new trade
        
        Args:
            signal: 'BUY' or 'SELL'
            entry_price: Entry price
            entry_time: Entry timestamp
            entry_index: Bar index
            atr_value: ATR value (for ATR mode)
            
        Returns:
            Trade object or None if can't open
        """
        # Check max concurrent trades
        if len(self.open_trades) >= self.config['max_concurrent_trades']:
            return None
        
        # Apply spread
        if signal == 'BUY':
            actual_entry = entry_price + (self.spread_pips * self.pip_value)
        else:
            actual_entry = entry_price - (self.spread_pips * self.pip_value)
        
        # Calculate SL/TP
        sl_price, tp_price, sl_pips, tp_pips = self.calculate_sl_tp(
            signal, actual_entry, atr_value
        )
        
        # Calculate position size
        lot = self.calculate_lot_size(sl_pips)
        
        # Calculate commission
        commission = self.config['commission_per_lot'] * lot
        
        # Create trade
        trade = Trade(
            trade_id=self.trade_counter,
            signal=signal,
            entry_time=entry_time,
            entry_price=actual_entry,
            entry_index=entry_index,
            lot_size=lot,
            sl_price=sl_price,
            tp_price=tp_price,
            commission=commission
        )
        
        self.trade_counter += 1
        self.trades.append(trade)
        self.open_trades.append(trade)
        
        return trade
    
    def check_exit(self, trade: Trade, high: float, low: float, 
                   close: float, timestamp: pd.Timestamp, index: int) -> bool:
        """
        Check if trade should be exited
        
        Args:
            trade: Trade object
            high: Bar high
            low: Bar low
            close: Bar close
            timestamp: Bar timestamp
            index: Bar index
            
        Returns:
            True if trade was closed
        """
        if not trade.is_open:
            return False
        
        exit_price = None
        exit_reason = None
        
        if trade.signal == 'BUY':
            # Check SL hit
            if low <= trade.sl_price:
                exit_price = trade.sl_price
                exit_reason = 'SL'
            # Check TP hit
            elif high >= trade.tp_price:
                exit_price = trade.tp_price
                exit_reason = 'TP'
        else:  # SELL
            # Check SL hit
            if high >= trade.sl_price:
                exit_price = trade.sl_price
                exit_reason = 'SL'
            # Check TP hit
            elif low <= trade.tp_price:
                exit_price = trade.tp_price
                exit_reason = 'TP'
        
        if exit_price is not None:
            self.close_trade(trade, exit_price, timestamp, index, exit_reason)
            return True
        
        return False
    
    def close_trade(self, trade: Trade, exit_price: float, 
                    exit_time: pd.Timestamp, exit_index: int, 
                    exit_reason: str):
        """Close a trade"""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_index = exit_index
        trade.exit_reason = exit_reason
        trade.is_open = False
        
        # Calculate P&L
        trade.calculate_pnl(exit_price, self.pip_value, self.lot_size)
        
        # Update capital
        self.current_capital += trade.pnl
        
        # Remove from open trades
        if trade in self.open_trades:
            self.open_trades.remove(trade)
    
    def close_all_trades(self, close_price: float, timestamp: pd.Timestamp, index: int):
        """Close all open trades at current price"""
        for trade in list(self.open_trades):
            self.close_trade(trade, close_price, timestamp, index, 'CLOSE')
    
    def update_equity(self, timestamp: pd.Timestamp):
        """Update equity curve"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.current_capital
        })
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of metrics
        """
        closed_trades = [t for t in self.trades if not t.is_open]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
            }
        
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in closed_trades)
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        win_rate = (len(winning_trades) / len(closed_trades)) * 100
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()
            max_drawdown_pct = (max_drawdown / equity_df['peak'].max()) * 100
        else:
            max_drawdown = 0
            max_drawdown_pct = 0
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
        }
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """Export trades to DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'signal': trade.signal,
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason,
                'lot_size': trade.lot_size,
                'sl_price': trade.sl_price,
                'tp_price': trade.tp_price,
                'pnl': trade.pnl,
                'pnl_pips': trade.pnl_pips,
                'commission': trade.commission,
                'is_open': trade.is_open,
            })
        
        return pd.DataFrame(trades_data)
