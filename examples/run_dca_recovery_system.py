"""
Smart DCA Recovery System - Complete Example

Demonstrates the full Smart DCA Recovery System:
1. Statistical analysis of price movements
2. Intelligent grid calculation
3. Reversal prediction (rule-based or ML)
4. DCA position management
5. Performance comparison vs traditional single-entry recovery

Usage:
    python examples/run_dca_recovery_system.py

Config:
    Edit config.py -> DCA_RECOVERY_CONFIG to customize settings

Author: Claude Code
Date: 2025-10-27
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
from datetime import datetime

# Import DCA recovery modules
from core.recovery.statistical_analyzer import StatisticalAnalyzer
from core.recovery.reversal_predictor import ReversalPredictor
from core.recovery.grid_calculator import GridCalculator
from core.recovery.dca_position_manager import DCAPositionManager

from config import DATA_DIR


# Default configuration
DEFAULT_CONFIG = {
    # Data settings
    'symbol': 'GBPUSD',
    'timeframe': 'M15',
    'days': 180,

    # Mode selection
    'reversal_mode': 'rule-based',  # 'rule-based' or 'ml'

    # Grid settings
    'n_grid_levels': 5,
    'lot_multiplier': 1.5,  # Each level = previous × 1.5
    'max_risk_usd': 500,
    'spacing_mode': 'statistical',  # 'statistical', 'atr', or 'hybrid'

    # Entry settings
    'initial_lot_size': 0.1,
    'min_profit_pips': 20,  # Min profit to close position

    # Trading settings
    'pip_value': 0.0001,
    'pip_value_in_usd': 10.0,
    'commission_per_lot': 7.0,
}


def print_config(config):
    """Print configuration"""
    print("\n" + "="*120)
    print("SMART DCA RECOVERY SYSTEM - CONFIGURATION")
    print("="*120)
    print(f"\n[DATA]")
    print(f"  Symbol: {config['symbol']}")
    print(f"  Timeframe: {config['timeframe']}")
    print(f"  History: {config['days']} days")

    print(f"\n[MODE]")
    print(f"  Reversal Prediction: {config['reversal_mode'].upper()}")

    print(f"\n[GRID]")
    print(f"  Levels: {config['n_grid_levels']}")
    print(f"  Lot Multiplier: {config['lot_multiplier']}x")
    print(f"  Max Risk: ${config['max_risk_usd']}")
    print(f"  Spacing Mode: {config['spacing_mode']}")

    print(f"\n[TRADING]")
    print(f"  Initial Lot: {config['initial_lot_size']}")
    print(f"  Min Profit: {config['min_profit_pips']} pips")
    print(f"  Commission: ${config['commission_per_lot']}/lot")
    print("="*120)


def run_statistical_analysis(data, config):
    """Run statistical analysis on data"""
    print("\n" + "="*120)
    print("STEP 1: STATISTICAL ANALYSIS")
    print("="*120)

    analyzer = StatisticalAnalyzer(data, pip_value=config['pip_value'])
    analyzer.print_analysis()

    return analyzer


def initialize_components(data, stat_analyzer, config):
    """Initialize all components"""
    print("\n" + "="*120)
    print("STEP 2: INITIALIZE COMPONENTS")
    print("="*120)

    # Reversal predictor
    print(f"\nInitializing {config['reversal_mode']} Reversal Predictor...")
    rev_predictor = ReversalPredictor(mode=config['reversal_mode'])
    rev_predictor.initialize(data, stat_analyzer)

    if config['reversal_mode'] == 'ml':
        print("Training ML model...")
        rev_predictor.train(data)

    print(f"OK {config['reversal_mode']} predictor ready")

    # Grid calculator
    print("\nInitializing Grid Calculator...")
    grid_calculator = GridCalculator(
        statistical_analyzer=stat_analyzer,
        reversal_predictor=rev_predictor,
        pip_value=config['pip_value'],
        pip_value_in_usd=config['pip_value_in_usd'],
        commission_per_lot=config['commission_per_lot']
    )
    print("OK Grid Calculator ready")

    # Position manager
    print("\nInitializing DCA Position Manager...")
    position_manager = DCAPositionManager(
        grid_calculator=grid_calculator,
        reversal_predictor=rev_predictor,
        pip_value=config['pip_value'],
        pip_value_in_usd=config['pip_value_in_usd'],
        commission_per_lot=config['commission_per_lot']
    )
    print("OK Position Manager ready")

    return rev_predictor, grid_calculator, position_manager


def demonstrate_grid_calculation(grid_calculator, data, config):
    """Demonstrate grid calculation for sample scenarios"""
    print("\n" + "="*120)
    print("STEP 3: GRID CALCULATION EXAMPLES")
    print("="*120)

    # Example 1: SELL trade that went wrong
    print("\n[EXAMPLE 1] SELL @ 1.2500, price moved up to 1.2650 (150 pips adverse)")
    print("-" * 120)

    grid = grid_calculator.calculate_grid(
        entry_price=1.2500,
        entry_lot=config['initial_lot_size'],
        direction='sell',
        current_idx=5000,
        n_levels=config['n_grid_levels'],
        lot_multiplier=config['lot_multiplier'],
        max_risk_usd=config['max_risk_usd'],
        spacing_mode=config['spacing_mode']
    )

    grid_calculator.print_grid(grid, "SELL Recovery Grid (Statistical Spacing)")

    # Example 2: BUY trade that went wrong
    print("\n[EXAMPLE 2] BUY @ 1.2700, price moved down to 1.2550 (150 pips adverse)")
    print("-" * 120)

    grid = grid_calculator.calculate_grid(
        entry_price=1.2700,
        entry_lot=config['initial_lot_size'],
        direction='buy',
        current_idx=5000,
        n_levels=config['n_grid_levels'],
        lot_multiplier=config['lot_multiplier'],
        max_risk_usd=config['max_risk_usd'],
        spacing_mode=config['spacing_mode']
    )

    grid_calculator.print_grid(grid, "BUY Recovery Grid (Statistical Spacing)")


def simulate_losing_streak_recovery(position_manager, data, config):
    """
    Simulate a losing streak scenario and show how DCA recovery works

    Scenario:
    - Initial trade: SELL @ 1.2500 (loses money)
    - Price moves against us up to 1.2650
    - DCA system adds entries according to grid
    - Price eventually reverts
    - System closes for profit
    """
    print("\n" + "="*120)
    print("STEP 4: SIMULATE LOSING STREAK RECOVERY")
    print("="*120)

    print("\n[SCENARIO] Trader sold GBPUSD @ 1.2500, but price moved UP against the trade")
    print("-" * 120)

    # Find a suitable starting point (price around 1.2500)
    target_price = 1.2500
    start_idx = (data['close'] - target_price).abs().idxmin()
    start_idx_pos = data.index.get_loc(start_idx)

    # Ensure enough history
    if start_idx_pos < 1000:
        start_idx_pos = 1000

    entry_price = data['close'].iloc[start_idx_pos]
    entry_timestamp = data.index[start_idx_pos]

    print(f"\n[T+0] Opening SELL position:")
    print(f"      Time: {entry_timestamp}")
    print(f"      Entry: {entry_price:.5f}")
    print(f"      Lot: {config['initial_lot_size']}")

    # Open position
    position = position_manager.open_position(
        symbol=config['symbol'],
        direction='SELL',
        entry_price=entry_price,
        entry_lot=config['initial_lot_size'],
        entry_timestamp=entry_timestamp,
        current_idx=start_idx_pos,
        n_grid_levels=config['n_grid_levels'],
        lot_multiplier=config['lot_multiplier'],
        max_risk_usd=config['max_risk_usd']
    )

    print(f"\n[GRID] Calculated grid levels:")
    for level in position.grid_levels:
        print(f"      {level}")

    # Simulate price movement
    print(f"\n[SIMULATION] Monitoring price movement (next 500 bars)...")
    print("-" * 120)

    max_bars = min(500, len(data) - start_idx_pos - 1)
    events = []

    for i in range(1, max_bars):
        current_idx = start_idx_pos + i
        current_price = data['close'].iloc[current_idx]
        current_timestamp = data.index[current_idx]

        # Update position
        prev_entries = len(position.entries)
        position_manager.update(current_price, current_timestamp, current_idx)

        # Check if new entry added
        if len(position.entries) > prev_entries:
            events.append({
                'bar': i,
                'timestamp': current_timestamp,
                'type': 'DCA_ENTRY',
                'price': current_price,
                'details': f"Added Level {position.next_grid_level - 1}"
            })

        # Check if should close
        if position.should_close(current_price, config['pip_value'], config['pip_value_in_usd'], config['min_profit_pips']):
            position_manager.close_profitable_positions(current_price, current_timestamp, config['min_profit_pips'])
            events.append({
                'bar': i,
                'timestamp': current_timestamp,
                'type': 'CLOSE',
                'price': current_price,
                'details': f"Position closed with profit"
            })
            break

    # Print events
    print(f"\n[EVENTS] Key events during simulation:")
    for event in events:
        print(f"  Bar {event['bar']:4d} | {event['timestamp']} | {event['type']:12s} | {event['price']:.5f} | {event['details']}")

    # Print final position status
    print(f"\n[RESULT]")
    if position.status.value == 'CLOSED':
        print(f"  Status: CLOSED")
        print(f"  Total Entries: {len(position.entries)}")
        print(f"  Average Price: {position.average_entry_price:.5f}")
        print(f"  Exit Price: {position.exit_price:.5f}")
        print(f"  Realized P&L: ${position.realized_pnl:.2f}")
        print(f"  Max Risk Was: ${position.grid_levels[-1].risk_usd if position.grid_levels else 0:.2f}")

        if position.realized_pnl > 0:
            print(f"\n  ✅ SUCCESS: Recovered from losing trade!")
        else:
            print(f"\n  ❌ LOSS: Position closed at loss")
    else:
        position.update_pnl(data['close'].iloc[start_idx_pos + max_bars - 1], config['pip_value'], config['pip_value_in_usd'])
        print(f"  Status: STILL OPEN")
        print(f"  Total Entries: {len(position.entries)}")
        print(f"  Average Price: {position.average_entry_price:.5f}")
        print(f"  Unrealized P&L: ${position.unrealized_pnl:.2f}")
        print(f"\n  ⏳ Position still waiting for reversion")


def main():
    """Main function"""

    config = DEFAULT_CONFIG

    print_config(config)

    # Load data
    print("\n" + "="*120)
    print("LOADING DATA")
    print("="*120)

    data_file = DATA_DIR / f"{config['symbol']}_{config['timeframe']}_{config['days']}days.csv"

    if not data_file.exists():
        print(f"\n❌ ERROR: Data file not found: {data_file}")
        print("\nPlease download data first:")
        print("   python data/batch_download_mt5_data.py")
        return 1

    print(f"\nLoading: {data_file}")
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"✅ Loaded {len(data)} candles")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")

    # Step 1: Statistical Analysis
    stat_analyzer = run_statistical_analysis(data, config)

    # Step 2: Initialize Components
    rev_predictor, grid_calculator, position_manager = initialize_components(data, stat_analyzer, config)

    # Step 3: Demonstrate Grid Calculation
    demonstrate_grid_calculation(grid_calculator, data, config)

    # Step 4: Simulate Recovery
    simulate_losing_streak_recovery(position_manager, data, config)

    # Final Summary
    position_manager.print_summary()

    print("\n" + "="*120)
    print("DONE")
    print("="*120)
    print("\n[INFO] To switch between rule-based and ML mode:")
    print("       Edit config['reversal_mode'] = 'rule-based' or 'ml'")
    print("\n[INFO] To adjust grid settings:")
    print("       Edit config['n_grid_levels'], config['lot_multiplier'], etc.")
    print("\n[INFO] For full backtest, use: examples/run_dca_backtest.py (coming soon)")
    print("="*120 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
