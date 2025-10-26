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
    # MT5 executable path - THAY �I � CH�N MT5 N�O S� D�NG
    # Windows default paths:
    'path': r'C:\Users\nguye\AppData\Roaming\MetaTrader 5 IC Markets (SC)\terminal64.exe',
    # 'path': r'C:\Program Files\MetaTrader 5 - IC Markets\terminal64.exe',
    # 'path': r'C:\Program Files\MetaTrader 5 - XM\terminal64.exe',

    # Linux/Wine paths:
    # 'path': r'/home/user/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe',

    # Set to None to use default MT5 installation
    'path': None,

    # MT5 login credentials (optional - n�u mu�n auto login)
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
    # Default symbol and timeframe (for single download)
    'symbol': 'GBPUSD',
    'timeframe': 'M15',  # M1, M5, M15, M30, H1, H4, D1

    # Data range
    'days': 180,          # Download last N days
    'start_date': None,  # Or specify start date: '2024-01-01'
    'end_date': None,    # Or specify end date: '2024-12-31'

    # Output
    'save_csv': True,
    'csv_path': None,    # Auto-generate if None: data/{symbol}_{timeframe}_{days}days.csv
}


# ============================================
# BATCH DOWNLOAD CONFIGURATION
# ============================================
BATCH_DOWNLOAD_CONFIG = {
    # Enable/disable batch download
    'enabled': True,

    # Symbols to download (list of symbols)
    'symbols': [
        # Major pairs
         
        'GBPUSD',
         

        # Cross pairs (optional - comment out if not needed)
        # 'EURJPY',
        # 'GBPJPY',
        # 'EURGBP',
        # 'AUDJPY',

        # Commodities (optional)
        # 'XAUUSD',  # Gold
        # 'XAGUSD',  # Silver
        # 'USOIL',   # Crude Oil

        # Indices (optional)
        # 'US30',    # Dow Jones
        # 'NAS100',  # Nasdaq
        # 'SPX500',  # S&P 500
    ],

    # Timeframes to download for EACH symbol
    'timeframes': [
        
        'M15', 
        'H1',
        'H4',
        'D1',
        # 'M5',   # Uncomment if needed
        # 'M30',  # Uncomment if needed
        # 'D1',   # Uncomment if needed
    ],

    # Data range
    'days': 700,              # Download last 180 days

    # Download settings
    'skip_existing': True,    # Skip if file already exists
    'show_progress': True,    # Show progress bar
    'delay_between': 0.5,     # Delay between downloads (seconds)

    # Error handling
    'continue_on_error': True,  # Continue if one symbol fails
    'max_retries': 2,           # Retry N times if download fails
}


# ============================================
# FVG CONFIGURATION
# ============================================
FVG_CONFIG = {
    # FVG Detection
    'lookback_days': 90,           # FVG valid for 90 days
    'min_gap_atr_ratio': 0.5,      # Gap must be >= ATR x 0.5
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
# BACKTEST CONFIGURATION - DYNAMIC RISK RECOVERY SYSTEM
# ============================================
BACKTEST_CONFIG = {
    # ===== SYMBOL & TIMEFRAME =====
    'symbol': 'GBPUSD',
    'timeframe': 'M15',             # Base timeframe
    'fvg_timeframe': 'H1',         # FVG analysis timeframe
    'days': 700,                   # Number of days of data

    # ===== ACCOUNT SETTINGS =====
    'initial_balance': 1000.0,     # Starting balance
    'risk_per_trade': 0.02,        # Risk per trade (2%)
    'base_lot_size': 0.01,          # Base lot size in VIRTUAL mode

    # ===== COMMISSION & COSTS =====
    'commission_per_lot': 0.0,     # Commission per lot (round trip)
    'pip_value': 0.0001,           # For 5-digit broker (0.01 for 4-digit)
    'pip_value_in_account_currency': 10.0,  # $ per pip per standard lot (GBPUSD = $10)

    # ===== DYNAMIC RISK RECOVERY (Replaces Martingale!) =====
    'consecutive_losses_trigger': 3,  # Switch to REAL mode after N virtual losses
    'recovery_multiplier': 2.0,       # Recovery target = Total Loss × 2.0 (recover + profit)
    'min_lot_size': 0.01,             # Minimum lot size (broker limit)
    'max_lot_size': 10.0,             # Maximum lot size (risk limit)

    # ===== STOP LOSS / TAKE PROFIT MODE SELECTION =====
    # Choose: True = ATR-based (dynamic), False = Fixed pips
    'use_atr_sl_tp': True,        # ← Change to True for ATR mode

    # ATR Mode Settings (used when use_atr_sl_tp = True)
    'atr_sl_multiplier': 1.5,      # SL = ATR × 1.5
    'atr_tp_multiplier': 3.0,      # TP = ATR × 3.0

    # Pips Mode Settings (used when use_atr_sl_tp = False)
    'sl_pips': 50,                 # SL = 50 pips (fixed)
    'tp_pips': 100,                # TP = 100 pips (fixed)

    # ===== CONFLUENCE SCORING =====
    'min_confidence_score': 70.0,  # Minimum score to trade (70%)
    'enable_adx_filter': True,     # Enable ADX filter
    'adx_threshold': 35.0,         # ADX >= 35 for strong trends
    'use_statistical': True,       # Use statistical indicators (True) or basic (False)

    # ===== CONFLUENCE WEIGHTS =====
    # STATISTICAL MODE (use_statistical=True):
    # Total = 110, then -10 regime = 100
    'confluence_weights': {
        'fvg': 45,              # Primary signal (reduced, quality checked by fvg_size_atr)
        'fvg_size_atr': 15,     # FVG strength normalized by ATR - CRITICAL!
        'hurst': 10,            # Hurst Exponent (trend persistence)
        'lr_deviation': 20,     # Linear regression deviation - CRITICAL!
        'skewness': 10,         # Distribution bias (filter)
        'kurtosis': 5,          # Fat tails detection (filter)
        'obv_div': 5,          # OBV Divergence
        'overlap_count': 0,     # Multi-TF overlap (disabled by default, can enable for testing)
        'regime': -10,          # Market Regime penalty (negative!)
    },
    # BASIC MODE (use_statistical=False):
    # 'confluence_weights': {
    #     'fvg': 50,      # FVG weight (primary signal)
    #     'vwap': 20,     # VWAP weight
    #     'obv': 15,      # OBV weight
    #     'volume': 15,   # Volume spike weight
    # },

    # ===== BACKTEST LIMITS =====
    'max_trades_per_day': 50,      # Max trades per day
    'max_concurrent_trades': 1,    # Only 1 trade at a time
}


# ============================================
# LEGACY BACKTEST CONFIG (OLD - For reference only)
# ============================================
BACKTEST_CONFIG_LEGACY = {
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
# MULTI-TIMEFRAME STRATEGY CONFIGURATION
# ============================================
MULTI_TIMEFRAME_STRATEGY_CONFIG = {
    # Base timeframe (smallest timeframe for execution)
    'base_timeframe': 'M15',       # Trading execution timeframe

    # FVG Analysis Timeframes
    'fvg_timeframes': {
        'primary': 'H1',           # Primary FVG timeframe
        'secondary': 'H4',         # Secondary FVG timeframe (optional)
        'tertiary': None,          # Tertiary FVG timeframe (optional, e.g., 'D1')
    },

    # Indicators Timeframes
    'indicator_timeframes': {
        # Trend indicators
        'ema_fast': 'M15',         # Fast EMA timeframe
        'ema_slow': 'H1',          # Slow EMA timeframe
        'macd': 'M15',             # MACD timeframe

        # Momentum indicators
        'rsi': 'M15',              # RSI timeframe
        'stochastic': 'M15',       # Stochastic timeframe

        # Volatility indicators
        'atr': 'M15',              # ATR timeframe (for base execution)
        'bollinger': 'H1',         # Bollinger Bands timeframe

        # Volume indicators
        'volume_sma': 'M15',       # Volume SMA timeframe
        'obv': 'H1',               # OBV timeframe
        'cmf': 'M15',              # CMF timeframe
    },

    # Strategy Rules
    'require_all_fvg_alignment': False,  # True = All FVG timeframes must align
    'min_timeframe_confluence': 2,        # Min timeframes that must agree

    # Example Strategies (can enable/disable)
    'strategies': {
        'triple_timeframe': {
            'enabled': True,
            'description': 'H4 trend + H1 FVG + M15 entry',
            'fvg_tf': 'H1',
            'trend_tf': 'H4',
            'entry_tf': 'M15',
        },
        'dual_confirmation': {
            'enabled': False,
            'description': 'H1 FVG + M15 RSI',
            'fvg_tf': 'H1',
            'entry_tf': 'M15',
        },
    }
}


# ============================================
# TESTING CONFIGURATION
# ============================================
TESTING_CONFIG = {
    # Test data generation
    'test_data_candles': 500,
    'test_data_symbol': 'GBPUSD',
    'test_data_timeframe': 'H1',
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

    print("\n=� Paths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Data Dir: {DATA_DIR}")
    print(f"  Logs Dir: {LOGS_DIR}")

    print("\n=' MT5:")
    print(f"  Path: {MT5_CONFIG['path'] or 'Default'}")
    print(f"  Server: {MT5_CONFIG['server'] or 'Not set'}")

    print("\n=� Data:")
    print(f"  Symbol: {DATA_CONFIG['symbol']}")
    print(f"  Timeframe: {DATA_CONFIG['timeframe']}")
    print(f"  Days: {DATA_CONFIG['days']}")

    print("\n=� FVG:")
    print(f"  Lookback: {FVG_CONFIG['lookback_days']} days")
    print(f"  Min Gap ATR Ratio: {FVG_CONFIG['min_gap_atr_ratio']}")

    print("\n=� Strategy:")
    print(f"  Min Confluence: {STRATEGY_CONFIG['min_confluence_score']}")
    print(f"  SL: {STRATEGY_CONFIG['sl_pips']} pips")
    print(f"  TP: {STRATEGY_CONFIG['tp_pips']} pips")

    print("\n<� Backtest:")
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
