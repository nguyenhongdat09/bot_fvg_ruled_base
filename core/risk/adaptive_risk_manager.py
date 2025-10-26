"""
Adaptive Risk Management System

Dynamically adjusts risk based on current losing streak to:
1. Reduce drawdown during bad periods
2. Hard stop at maximum acceptable streak (default: 7 losses)
3. Preserve capital for recovery

Theory:
- Losing streaks are inevitable in trading (probability theory)
- Once in a streak, probability of continuation increases (clustering)
- Better to reduce exposure than fight the market
- Similar to Kelly Criterion but simpler and more conservative

NO OVERFITTING:
- Logic-based (not statistical fitting)
- No parameters tuned to historical data
- Works with ANY strategy/market
- Proven concept (used by institutional traders)

Author: Claude Code
Date: 2025-10-26
"""

from typing import List, Dict, Optional
from collections import deque


class AdaptiveRiskManager:
    """
    Adaptive Risk Management - Reduce risk during losing streaks

    Key Features:
    - Detects current losing streak length
    - Adjusts risk dynamically (100% → 75% → 50% → 0%)
    - Hard stop at max acceptable streak
    - Preserves capital during bad periods

    Usage:
        manager = AdaptiveRiskManager(max_lookback=15, max_acceptable_streak=7)

        # After each trade
        manager.update(is_win=True)  # or False

        # Before entering new trade
        risk_multiplier = manager.get_risk_multiplier()
        if risk_multiplier == 0:
            # Skip trade (in emergency mode)
            pass
        else:
            # Adjust lot size
            lot_size = base_lot_size * risk_multiplier
    """

    def __init__(self,
                 max_lookback: int = 15,
                 max_acceptable_streak: int = 7,
                 risk_reduction_schedule: Optional[Dict[int, float]] = None):
        """
        Initialize Adaptive Risk Manager

        Args:
            max_lookback: Number of recent trades to track (default: 15)
            max_acceptable_streak: Maximum losing streak before stopping (default: 7)
            risk_reduction_schedule: Custom risk schedule {streak_length: multiplier}
                                    Default: {0-2: 1.0, 3-4: 0.75, 5-6: 0.50, 7+: 0.0}
        """
        self.max_lookback = max_lookback
        self.max_acceptable_streak = max_acceptable_streak

        # Trade history (1 = win, 0 = loss)
        self.trade_history = deque(maxlen=max_lookback)

        # Risk reduction schedule
        if risk_reduction_schedule is None:
            # Default conservative schedule
            self.risk_schedule = {
                0: 1.00,   # No streak: full risk
                1: 1.00,   # 1 loss: full risk
                2: 1.00,   # 2 losses: full risk
                3: 0.75,   # 3 losses: reduce 25%
                4: 0.75,   # 4 losses: reduce 25%
                5: 0.50,   # 5 losses: reduce 50%
                6: 0.50,   # 6 losses: reduce 50%
                7: 0.00,   # 7+ losses: STOP TRADING
            }
        else:
            self.risk_schedule = risk_reduction_schedule

        # Statistics tracking
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.max_streak_seen = 0
        self.times_stopped = 0  # How many times hit emergency stop

    def update(self, is_win: bool):
        """
        Update with latest trade result

        Args:
            is_win: True if trade was profitable, False if loss
        """
        # Record result (1 = win, 0 = loss)
        self.trade_history.append(1 if is_win else 0)

        # Update statistics
        self.total_trades += 1
        if is_win:
            self.total_wins += 1
        else:
            self.total_losses += 1

        # Track max streak
        current_streak = self.get_current_streak()
        if current_streak > self.max_streak_seen:
            self.max_streak_seen = current_streak

        # Track emergency stops
        if current_streak >= self.max_acceptable_streak:
            self.times_stopped += 1

    def get_current_streak(self) -> int:
        """
        Get current consecutive losing streak length

        Returns:
            int: Number of consecutive losses (0 if last trade was win)
        """
        if not self.trade_history:
            return 0

        streak = 0
        # Count from most recent backwards
        for result in reversed(self.trade_history):
            if result == 0:  # Loss
                streak += 1
            else:  # Win
                break

        return streak

    def get_risk_multiplier(self) -> float:
        """
        Get risk adjustment multiplier for current streak

        Returns:
            float: Risk multiplier (0.0 to 1.0)
                  0.0 = Skip trade (emergency stop)
                  0.5 = Half risk
                  0.75 = Reduce 25%
                  1.0 = Full risk (normal)
        """
        streak = self.get_current_streak()

        # Look up in schedule
        if streak >= self.max_acceptable_streak:
            return 0.0  # Emergency stop

        # Find appropriate multiplier
        return self.risk_schedule.get(streak, 0.0)

    def should_trade(self) -> bool:
        """
        Check if should take trade (not in emergency stop mode)

        Returns:
            bool: True if should trade, False if should skip
        """
        return self.get_risk_multiplier() > 0.0

    def get_status(self) -> Dict:
        """
        Get current risk manager status

        Returns:
            dict: Status information including:
                - current_streak: Current losing streak length
                - risk_multiplier: Current risk adjustment
                - should_trade: Whether to take trades
                - is_emergency_stop: Whether in emergency mode
                - statistics: Overall performance stats
        """
        current_streak = self.get_current_streak()
        risk_multiplier = self.get_risk_multiplier()

        return {
            'current_streak': current_streak,
            'risk_multiplier': risk_multiplier,
            'should_trade': risk_multiplier > 0.0,
            'is_emergency_stop': current_streak >= self.max_acceptable_streak,
            'statistics': {
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'total_losses': self.total_losses,
                'win_rate': self.total_wins / self.total_trades if self.total_trades > 0 else 0,
                'max_streak_seen': self.max_streak_seen,
                'times_stopped': self.times_stopped
            }
        }

    def reset(self):
        """Reset risk manager (clear history, keep statistics)"""
        self.trade_history.clear()

    def get_recommendation(self) -> str:
        """
        Get human-readable recommendation for current state

        Returns:
            str: Recommendation message
        """
        streak = self.get_current_streak()
        multiplier = self.get_risk_multiplier()

        if multiplier == 0.0:
            return f"🛑 EMERGENCY STOP: {streak} consecutive losses! Skip all trades until streak breaks."
        elif multiplier == 0.50:
            return f"⚠️  DEFENSIVE MODE: {streak} losses. Reduce risk to 50% of normal."
        elif multiplier == 0.75:
            return f"🟡 CAUTION MODE: {streak} losses. Reduce risk to 75% of normal."
        else:
            return f"✅ NORMAL MODE: No significant losing streak. Trade with full risk."

    def print_status(self):
        """Print current status to console"""
        status = self.get_status()
        stats = status['statistics']

        print("\n" + "="*80)
        print("ADAPTIVE RISK MANAGER STATUS")
        print("="*80)

        print(f"\n📊 CURRENT STATE:")
        print(f"   Losing Streak: {status['current_streak']}")
        print(f"   Risk Multiplier: {status['risk_multiplier']*100:.0f}%")
        print(f"   Should Trade: {'YES' if status['should_trade'] else 'NO (EMERGENCY STOP)'}")

        print(f"\n📈 STATISTICS:")
        print(f"   Total Trades: {stats['total_trades']}")
        print(f"   Wins: {stats['total_wins']} ({stats['win_rate']*100:.1f}%)")
        print(f"   Losses: {stats['total_losses']}")
        print(f"   Max Streak Seen: {stats['max_streak_seen']}")
        print(f"   Emergency Stops: {stats['times_stopped']}")

        print(f"\n💡 RECOMMENDATION:")
        print(f"   {self.get_recommendation()}")

        print("="*80 + "\n")


class ConservativeRiskManager(AdaptiveRiskManager):
    """
    Very conservative risk manager (stops earlier, reduces more aggressively)

    Schedule:
    - 0-1 losses: 100% risk
    - 2-3 losses: 75% risk
    - 4-5 losses: 50% risk
    - 6+ losses: STOP
    """

    def __init__(self, max_lookback: int = 15):
        super().__init__(
            max_lookback=max_lookback,
            max_acceptable_streak=6,  # Stop at 6 instead of 7
            risk_reduction_schedule={
                0: 1.00,
                1: 1.00,
                2: 0.75,  # Start reducing earlier
                3: 0.75,
                4: 0.50,
                5: 0.50,
                6: 0.00,  # Stop at 6
            }
        )


class AggressiveRiskManager(AdaptiveRiskManager):
    """
    More aggressive risk manager (tolerates longer streaks)

    Schedule:
    - 0-3 losses: 100% risk
    - 4-5 losses: 80% risk
    - 6-7 losses: 60% risk
    - 8-9 losses: 40% risk
    - 10+ losses: STOP

    WARNING: Higher risk of large drawdowns!
    """

    def __init__(self, max_lookback: int = 15):
        super().__init__(
            max_lookback=max_lookback,
            max_acceptable_streak=10,  # Tolerate up to 10
            risk_reduction_schedule={
                0: 1.00,
                1: 1.00,
                2: 1.00,
                3: 1.00,
                4: 0.80,  # Less aggressive reduction
                5: 0.80,
                6: 0.60,
                7: 0.60,
                8: 0.40,
                9: 0.40,
                10: 0.00,
            }
        )


# Example usage
if __name__ == '__main__':
    print("\n" + "="*80)
    print("ADAPTIVE RISK MANAGER - EXAMPLE")
    print("="*80)

    # Create manager
    manager = AdaptiveRiskManager(max_acceptable_streak=7)

    # Simulate some trades
    print("\n[Simulating trades...]")

    # Start with some wins
    for i in range(3):
        manager.update(is_win=True)
        print(f"Trade {i+1}: WIN - Risk: {manager.get_risk_multiplier()*100:.0f}%")

    # Then a losing streak
    for i in range(8):
        manager.update(is_win=False)
        multiplier = manager.get_risk_multiplier()
        should_trade = "TRADE" if multiplier > 0 else "SKIP (EMERGENCY STOP)"
        print(f"Trade {i+4}: LOSS - Streak: {manager.get_current_streak()} - Risk: {multiplier*100:.0f}% - {should_trade}")

    # Print final status
    manager.print_status()

    # Compare different risk profiles
    print("\n" + "="*80)
    print("COMPARISON: CONSERVATIVE vs DEFAULT vs AGGRESSIVE")
    print("="*80)

    managers = {
        'Conservative': ConservativeRiskManager(),
        'Default': AdaptiveRiskManager(),
        'Aggressive': AggressiveRiskManager()
    }

    print(f"\n{'Streak':>10} | {'Conservative':>15} | {'Default':>15} | {'Aggressive':>15}")
    print("-" * 80)

    for streak in range(12):
        # Simulate streak
        for name, mgr in managers.items():
            mgr.trade_history.clear()
            for _ in range(streak):
                mgr.trade_history.append(0)

        risks = {name: mgr.get_risk_multiplier() for name, mgr in managers.items()}

        print(f"{streak:10d} | {risks['Conservative']*100:13.0f}% | {risks['Default']*100:13.0f}% | {risks['Aggressive']*100:13.0f}%")
