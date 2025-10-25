"""
Configuration file for FVG Bot
"""

# MetaTrader5 Configuration
MT5_CONFIG = {
    'login': None,  # Set your MT5 login
    'password': None,  # Set your MT5 password
    'server': None,  # Set your MT5 server
    'timeout': 60000,  # Connection timeout in ms
    'portable': False  # Use portable mode
}

# Trading Symbols
SYMBOLS = {
    'EURUSD': {
        'digits': 5,
        'point': 0.00001,
        'tick_size': 0.00001,
        'tick_value': 1.0
    },
    'GBPUSD': {
        'digits': 5,
        'point': 0.00001,
        'tick_size': 0.00001,
        'tick_value': 1.0
    },
    'USDJPY': {
        'digits': 3,
        'point': 0.001,
        'tick_size': 0.001,
        'tick_value': 1.0
    }
}

# Timeframes
TIMEFRAMES = {
    'M1': 1,      # 1 minute
    'M5': 5,      # 5 minutes
    'M15': 15,    # 15 minutes
    'M30': 30,    # 30 minutes
    'H1': 60,     # 1 hour
    'H4': 240,    # 4 hours
    'D1': 1440    # 1 day
}

# FVG Detection Parameters
FVG_CONFIG = {
    'lookback_days': 90,
    'min_gap_atr_ratio': 0.3,
    'min_gap_pips': None,  # Optional minimum gap in pips
    'atr_period': 14
}

# Indicator Parameters
INDICATOR_CONFIG = {
    'ema_periods': [20, 50, 200],
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'atr_period': 14,
    'volume_ma_period': 20,
    'bollinger_period': 20,
    'bollinger_std': 2
}

# Backtest Configuration
BACKTEST_CONFIG = {
    'initial_balance': 10000,
    'risk_per_trade': 0.01,  # 1% risk per trade
    'commission': 0.0,
    'slippage': 0,
    'max_positions': 1
}

# Output Directories
OUTPUT_CONFIG = {
    'data_dir': 'data',
    'logs_dir': 'logs',
    'charts_dir': 'logs/charts',
    'results_dir': 'logs/results'
}
