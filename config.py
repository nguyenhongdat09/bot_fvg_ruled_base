"""
Configuration file for FVG Trading Bot

Chứa các cấu hình cho:
- Backtest
- SL/TP modes (Fixed pips hoặc ATR-based)
- Risk management
- FVG detection
"""

# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================

BACKTEST_CONFIG = {
    # === Data Settings ===
    'symbol': 'EURUSD',
    'timeframe': 'M15',  # M1, M5, M15, M30, H1, H4, D1
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    
    # === Initial Capital ===
    'initial_capital': 10000.0,
    'risk_per_trade_percent': 1.0,  # Risk 1% per trade
    
    # === SL/TP Mode Selection ===
    # 'fixed' = Use fixed pip values
    # 'atr' = Use ATR multiplier
    'sl_tp_mode': 'fixed',  # Options: 'fixed', 'atr'
    
    # === Fixed SL/TP Settings (in pips) ===
    # Used when sl_tp_mode = 'fixed'
    'fixed_sl_pips': 20.0,
    'fixed_tp_pips': 40.0,
    
    # === ATR-based SL/TP Settings ===
    # Used when sl_tp_mode = 'atr'
    'atr_period': 14,
    'atr_sl_multiplier': 1.5,  # SL = ATR × 1.5
    'atr_tp_multiplier': 3.0,  # TP = ATR × 3.0
    
    # === FVG Settings ===
    'fvg_lookback_days': 90,
    'fvg_min_gap_atr_ratio': 0.3,  # Minimum gap size = ATR × 0.3
    'fvg_min_gap_pips': None,  # Optional minimum gap in pips
    
    # === Trade Management ===
    'use_trailing_stop': False,
    'trailing_stop_pips': 15.0,
    'max_concurrent_trades': 1,
    
    # === Commission & Spread ===
    'commission_per_lot': 0.0,  # Commission per lot
    'spread_pips': 1.0,  # Spread in pips
    
    # === Symbol Info ===
    'pip_value': 0.0001,  # For EURUSD, 1 pip = 0.0001
    'lot_size': 100000,  # Standard lot
    'min_lot': 0.01,
    'max_lot': 100.0,
}

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

STRATEGY_CONFIG = {
    # === Signal Generation ===
    'use_fvg_confluence': True,
    'min_confluence_score': 0.6,
    
    # === FVG Target Settings ===
    'use_fvg_tp_target': False,  # Use FVG zone as TP target instead of fixed TP
    'fvg_tp_zone_percent': 0.5,  # Target 50% of FVG zone
    
    # === Entry Filters ===
    'require_trend_alignment': False,
    'require_volume_confirmation': False,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_CONFIG = {
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_file_path': 'logs/backtest.log',
    'log_trades': True,
    'log_trades_path': 'logs/trades.csv',
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VIZ_CONFIG = {
    'save_charts': True,
    'charts_dir': 'logs/charts',
    'show_fvgs': True,
    'show_trades': True,
    'show_equity_curve': True,
}
