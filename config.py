# config.py
"""
Configuration file for FVG Trading Bot
Centralized settings for all modules
"""

import os
from pathlib import Path

# ============================================
# PROJECT PATHS
# ============================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
LOGS_DIR = PROJECT_ROOT / 'logs'
CHARTS_DIR = LOGS_DIR / 'charts'

# Create directories if not exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)


# ============================================
# METATRADER 5 CONFIGURATION
# ============================================
MT5_CONFIG = {
    # MT5 executable path - THAY ÔI Â CHÌN MT5 NÀO Sì DäNG
    # Windows default paths:
    # 'path': r'C:\Program Files\MetaTrader 5\terminal64.exe',
    # 'path': r'C:\Program Files\MetaTrader 5 - IC Markets\terminal64.exe',
    # 'path': r'C:\Program Files\MetaTrader 5 - XM\terminal64.exe',

    # Linux/Wine paths:
    # 'path': r'/home/user/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe',

    # Set to None to use default MT5 installation
    'path': None,

    # MT5 login credentials (optional - n¿u muÑn auto login)
    'login': None,      # Account number
    'password': None,   # Password
    'server': None,     # Server name (e.g., 'ICMarkets-Demo')

    # Timeout settings
    'timeout': 60000,   # Connection timeout (ms)
}


# ============================================
# DATA DOWNLOAD CONFIGURATION
# ============================================
DATA_CONFIG = {
    # Default symbol and timeframe
    'symbol': 'EURUSD',
    'timeframe': 'M15',  # M1, M5, M15, M30, H1, H4, D1

    # Data range
    'days': 30,          # Download last N days
    'start_date': None,  # Or specify start date: '2024-01-01'
    'end_date': None,    # Or specify end date: '2024-12-31'

    # Output
    'save_csv': True,
    'csv_path': None,    # Auto-generate if None: data/{symbol}_{timeframe}_{days}days.csv
}


# ============================================
# FVG CONFIGURATION
# ============================================
FVG_CONFIG = {
    # FVG Detection
    'lookback_days': 90,           # FVG valid for 90 days
    'min_gap_atr_ratio': 0.3,      # Gap must be >= ATR × 0.3
    'min_gap_pips': None,          # Optional: Gap must be >= X pips

    # Visualization
    'show_touched_fvgs': True,     # Show touched FVGs in charts
    'show_labels': True,           # Show labels on FVG zones
    'chart_template': 'plotly_dark',  # Chart theme
}


# ============================================
# INDICATORS CONFIGURATION
# ============================================
INDICATORS_CONFIG = {
    # ATR (Average True Range)
    'atr_period': 14,
    'min_atr_threshold': 0.00010,  # Minimum ATR for trading (10 pips)

    # ADX (Average Directional Index)
    'adx_period': 14,
    'adx_threshold': 25,           # ADX >= 25 for trending market

    # RSI (Relative Strength Index)
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,

    # MACD
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    # Bollinger Bands
    'bb_period': 20,
    'bb_std': 2,

    # OBV (On-Balance Volume)
    'obv_divergence_window': 14,

    # CMF (Chaikin Money Flow)
    'cmf_period': 20,
    'cmf_threshold': 0.05,         # CMF > 0.05 strong buying
}


# ============================================
# STRATEGY CONFIGURATION
# ============================================
STRATEGY_CONFIG = {
    # Confluence scoring
    'min_confluence_score': 7,     # Minimum score to trade
    'high_confidence_score': 10,   # High confidence threshold

    # Position sizing
    'base_risk_percent': 1.0,      # Risk 1% per trade (medium confidence)
    'high_confidence_risk': 2.0,   # Risk 2% (high confidence)

    # Stop Loss / Take Profit (in pips)
    'sl_pips': 50,
    'tp_pips': 100,

    # Or use ATR-based SL/TP
    'use_atr_sl_tp': False,
    'sl_atr_multiplier': 1.5,
    'tp_atr_multiplier': 3.0,
}


# ============================================
# BACKTEST CONFIGURATION
# ============================================
BACKTEST_CONFIG = {
    # Virtual/Real mode switching
    'loss_streak_trigger': 3,      # Switch to REAL after N losses

    # Martingale settings
    'base_lot': 0.01,
    'martingale_factor': 1.3,      # Lot × 1.3 after each loss
    'max_martingale_steps': 5,     # Max 5 consecutive real trades

    # Starting capital
    'initial_balance': 10000,      # USD

    # Max trades
    'max_trades_per_day': 10,
    'max_concurrent_trades': 1,

    # Timeout
    'max_bars_in_trade': 50,       # Exit if no SL/TP hit after 50 candles
}


# ============================================
# LOGGING CONFIGURATION
# ============================================
LOGGING_CONFIG = {
    'level': 'INFO',               # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_to_console': True,
    'log_file': LOGS_DIR / 'trading_bot.log',

    # Trade logs
    'save_virtual_trades': True,
    'save_real_trades': True,
    'virtual_trades_csv': LOGS_DIR / 'virtual_trades.csv',
    'real_trades_csv': LOGS_DIR / 'real_trades.csv',

    # FVG logs
    'save_fvg_history': True,
    'fvg_history_csv': LOGS_DIR / 'fvg_history.csv',
}


# ============================================
# TESTING CONFIGURATION
# ============================================
TESTING_CONFIG = {
    # Test data generation
    'test_data_candles': 500,
    'test_data_symbol': 'EURUSD',
    'test_data_timeframe': 'M15',
    'test_random_seed': 42,        # For reproducible results

    # Visualization
    'create_test_charts': True,
    'test_charts_dir': CHARTS_DIR / 'test',
}


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_mt5_path():
    """
    Get MT5 executable path

    Returns:
        str or None: MT5 path
    """
    return MT5_CONFIG.get('path')


def get_data_filepath(symbol=None, timeframe=None, days=None):
    """
    Generate data file path

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        days: Number of days

    Returns:
        Path: File path
    """
    symbol = symbol or DATA_CONFIG['symbol']
    timeframe = timeframe or DATA_CONFIG['timeframe']
    days = days or DATA_CONFIG['days']

    filename = f"{symbol}_{timeframe}_{days}days.csv"
    return DATA_DIR / filename


def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)

    print("\n=Á Paths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Logs Dir: {LOGS_DIR}")

    print("\n=' MT5:")
    print(f"  Path: {MT5_CONFIG['path'] or 'Default'}")
    print(f"  Server: {MT5_CONFIG['server'] or 'Not set'}")

    print("\n=Ê Data:")
    print(f"  Symbol: {DATA_CONFIG['symbol']}")
    print(f"  Timeframe: {DATA_CONFIG['timeframe']}")
    print(f"  Days: {DATA_CONFIG['days']}")

    print("\n=Ð FVG:")
    print(f"  Lookback: {FVG_CONFIG['lookback_days']} days")
    print(f"  Min Gap ATR Ratio: {FVG_CONFIG['min_gap_atr_ratio']}")

    print("\n=È Strategy:")
    print(f"  Min Confluence: {STRATEGY_CONFIG['min_confluence_score']}")
    print(f"  SL: {STRATEGY_CONFIG['sl_pips']} pips")
    print(f"  TP: {STRATEGY_CONFIG['tp_pips']} pips")

    print("\n<² Backtest:")
    print(f"  Loss Streak Trigger: {BACKTEST_CONFIG['loss_streak_trigger']}")
    print(f"  Base Lot: {BACKTEST_CONFIG['base_lot']}")
    print(f"  Martingale Factor: {BACKTEST_CONFIG['martingale_factor']}")

    print("="*60)


# ============================================
# QUICK ACCESS VARIABLES
# ============================================

# MT5
MT5_PATH = MT5_CONFIG['path']
MT5_LOGIN = MT5_CONFIG['login']
MT5_PASSWORD = MT5_CONFIG['password']
MT5_SERVER = MT5_CONFIG['server']

# FVG
FVG_LOOKBACK_DAYS = FVG_CONFIG['lookback_days']
MIN_GAP_ATR_RATIO = FVG_CONFIG['min_gap_atr_ratio']

# Strategy
MIN_CONFLUENCE = STRATEGY_CONFIG['min_confluence_score']
SL_PIPS = STRATEGY_CONFIG['sl_pips']
TP_PIPS = STRATEGY_CONFIG['tp_pips']


if __name__ == '__main__':
    # Print config when run directly
    print_config()
