"""
FVG + Confluence Strategy

Strategy ket hop FVG voi ConfluenceScorer de ra tin hieu giao dich.
Khong dung RSI/MACD, chi dung:
- FVG (50%)
- VWAP (20%)
- OBV (15%)
- Volume Spike (15%)
- Optional ADX filter

Author: Claude Code
Date: 2025-10-25
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
from typing import Optional, Dict
from core.fvg.multi_timeframe_manager import MultiTimeframeManager
from indicators.volatility import ATRIndicator
from indicators.volume import VWAPIndicator, OBVIndicator, VolumeAnalyzer
from indicators.trend import ADXIndicator
from indicators.confluence import ConfluenceScorer
from config import MULTI_TIMEFRAME_STRATEGY_CONFIG


class FVGConfluenceStrategy:
    """
    FVG + Confluence Strategy

    Strategy Flow:
    1. FVG provides PRIMARY signal (BULLISH/BEARISH bias)
    2. Indicators provide CONFLUENCE score (0-100%)
    3. Score >= 70% = HIGH confidence -> Trade
    4. Score 60-70% = MEDIUM confidence -> Trade (optional)
    5. Score < 60% = LOW confidence -> Skip

    Components:
    - FVG: 50% weight (trend direction)
    - VWAP: 20% weight (price position)
    - OBV: 15% weight (volume trend)
    - Volume Spike: 15% weight (momentum)
    - ADX: Optional filter (remove choppy markets)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        base_timeframe: str = 'M15',
        fvg_timeframe: str = 'H1',
        config: dict = None,
        enable_adx_filter: bool = True,
        adx_threshold: float = 25.0,
        min_score_threshold: float = 70.0
    ):
        """
        Initialize strategy

        Args:
            data: OHLCV data (base timeframe)
            base_timeframe: Base timeframe (e.g., 'M15')
            fvg_timeframe: FVG analysis timeframe (e.g., 'H1')
            config: Custom config (default: MULTI_TIMEFRAME_STRATEGY_CONFIG)
            enable_adx_filter: Enable ADX filter
            adx_threshold: ADX threshold for trending market
            min_score_threshold: Minimum confluence score to trade (70%)
        """
        self.data = data.copy()
        self.base_timeframe = base_timeframe
        self.fvg_timeframe = fvg_timeframe
        self.config = config or MULTI_TIMEFRAME_STRATEGY_CONFIG

        self.enable_adx_filter = enable_adx_filter
        self.adx_threshold = adx_threshold
        self.min_score_threshold = min_score_threshold

        print(f"\n{'='*80}")
        print("FVG + CONFLUENCE STRATEGY INITIALIZATION")
        print(f"{'='*80}")
        print(f"Base Timeframe: {base_timeframe}")
        print(f"FVG Timeframe: {fvg_timeframe}")
        print(f"Data: {len(data)} candles")
        print(f"Date Range: {data.index[0]} to {data.index[-1]}")
        print(f"ADX Filter: {'Enabled' if enable_adx_filter else 'Disabled'}")
        print(f"Min Confluence Score: {min_score_threshold}%")

        # Setup components
        self._setup_fvg_manager()
        self._setup_indicators()
        self._setup_confluence_scorer()

        print(f"{'='*80}")
        print("[OK] INITIALIZATION COMPLETE")
        print(f"{'='*80}\n")

    def _setup_fvg_manager(self):
        """Setup FVG manager"""
        print("\n[FVG] Setting up FVG Manager...")
        print(f"   FVG Timeframe: {self.fvg_timeframe}")

        self.mtf_fvg = MultiTimeframeManager(
            self.data,
            base_timeframe=self.base_timeframe
        )
        self.mtf_fvg.add_fvg_timeframe(self.fvg_timeframe)

        print("   [OK] FVG Manager ready")

    def _setup_indicators(self):
        """Calculate indicators"""
        print("\n[INDICATORS] Calculating Indicators...")

        # ATR - for position sizing and SL/TP
        print("   Calculating ATR...")
        self.atr_indicator = ATRIndicator(period=14)
        self.data['atr'] = self.atr_indicator.calculate(self.data)

        # VWAP - 20% weight
        print("   Calculating VWAP...")
        self.vwap_indicator = VWAPIndicator()
        self.data['vwap'] = self.vwap_indicator.calculate(self.data)

        # OBV - 15% weight
        print("   Calculating OBV...")
        self.obv_indicator = OBVIndicator()
        self.data['obv'] = self.obv_indicator.calculate(self.data)

        # Volume Analyzer - 15% weight
        print("   Calculating Volume Spike...")
        self.volume_analyzer = VolumeAnalyzer(period=20)
        volume_data = self.volume_analyzer.calculate(self.data)
        self.data['avg_volume'] = volume_data['avg_volume']
        self.data['volume_ratio'] = volume_data['volume_ratio']
        self.data['is_spike'] = volume_data['is_spike']
        self.data['spike_strength'] = volume_data['spike_strength']

        # ADX - optional filter
        if self.enable_adx_filter:
            print("   Calculating ADX...")
            self.adx_indicator = ADXIndicator(period=14)
            adx_data = self.adx_indicator.calculate(self.data)
            self.data['adx'] = adx_data['adx']
            self.data['plus_di'] = adx_data['plus_di']
            self.data['minus_di'] = adx_data['minus_di']
            self.data['is_trending'] = self.adx_indicator.is_trending(
                self.data, threshold=self.adx_threshold
            )

        print("   [OK] All indicators calculated")

    def _setup_confluence_scorer(self):
        """Setup confluence scorer"""
        print("\n[CONFLUENCE] Setting up Confluence Scorer...")

        # Default weights: FVG 50%, VWAP 20%, OBV 15%, Volume 15%
        weights = {
            'fvg': 50,
            'vwap': 20,
            'obv': 15,
            'volume': 15,
        }

        self.confluence_scorer = ConfluenceScorer(
            weights=weights,
            adx_enabled=self.enable_adx_filter,
            adx_threshold=self.adx_threshold
        )

        print(f"   Weights: {weights}")
        print(f"   ADX Filter: {self.enable_adx_filter}")
        print("   [OK] Confluence Scorer ready")

    def analyze(self, index: int) -> Dict:
        """
        Analyze market at given index

        Args:
            index: Current index in base timeframe

        Returns:
            dict: Analysis result with signal and confluence score
        """
        # Update FVG manager
        self.mtf_fvg.update(index)

        # Get FVG structure
        fvg_structure = self.mtf_fvg.get_fvg_structure(self.fvg_timeframe, index)

        # Get ATR value
        atr_value = self.data.iloc[index]['atr']

        # Calculate confluence score
        confluence_result = self.confluence_scorer.calculate_score(
            data=self.data,
            index=index,
            fvg_structure=fvg_structure,
            atr_value=atr_value
        )

        # Add current price, timestamp, and FVG structure
        confluence_result['price'] = self.data.iloc[index]['close']
        confluence_result['timestamp'] = self.data.index[index]
        confluence_result['atr'] = atr_value
        confluence_result['fvg_structure'] = fvg_structure  # Add FVG structure for backtest logging

        return confluence_result

    def should_trade(self, analysis: Dict) -> bool:
        """
        Check if should trade based on analysis

        Args:
            analysis: Analysis result from analyze()

        Returns:
            bool: True if should trade
        """
        # Check signal exists
        if analysis['signal'] == 'NEUTRAL':
            return False

        # Check confluence score
        if analysis['total_score'] < self.min_score_threshold:
            return False

        # Check should_trade flag from confluence scorer
        if not analysis.get('should_trade', False):
            return False

        return True

    def get_data(self) -> pd.DataFrame:
        """Get strategy data with all indicators"""
        return self.data

    def get_fvg_manager(self) -> MultiTimeframeManager:
        """Get FVG manager"""
        return self.mtf_fvg


def quick_test():
    """Quick test function"""
    from config import DATA_DIR

    print("\n" + "="*80)
    print("FVG + CONFLUENCE STRATEGY TEST")
    print("="*80)

    # Load sample data
    print("\n1. Loading data...")
    data_file = DATA_DIR / "EURUSD_M15_180days.csv"

    if not data_file.exists():
        print(f"[ERROR] Data file not found: {data_file}")
        print("   Please run: python data/batch_download_mt5_data.py")
        return

    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print(f"   Loaded {len(data)} candles")

    # Initialize strategy
    print("\n2. Initializing strategy...")
    strategy = FVGConfluenceStrategy(
        data=data,
        base_timeframe='M15',
        fvg_timeframe='H1',
        enable_adx_filter=True,
        min_score_threshold=70.0
    )

    # Test analysis
    print("\n3. Testing analysis on recent candles...")
    start_idx = max(100, len(data) - 20)  # Last 20 candles

    signals = []
    for i in range(start_idx, len(data)):
        analysis = strategy.analyze(i)

        if analysis['signal'] != 'NEUTRAL' and analysis['total_score'] >= 70:
            signals.append(analysis)

            print(f"\n{'='*80}")
            print(f"[SIGNAL] {analysis['signal']} Signal at {analysis['timestamp']}")
            print(f"{'='*80}")
            print(f"Price: {analysis['price']:.5f}")
            print(f"Confluence Score: {analysis['total_score']:.1f}% ({analysis['confidence']})")
            print(f"\nComponents:")

            components = analysis['components']
            for comp_name, comp_data in components.items():
                if isinstance(comp_data, dict):
                    score = comp_data.get('score', 0)
                    status = comp_data.get('status', '')
                    print(f"  {comp_name.upper()}: {score:.1f}% - {status}")

            if 'sl_tp' in analysis:
                sl_tp = analysis['sl_tp']
                print(f"\nSL/TP Levels:")
                print(f"  SL: {sl_tp['sl']:.5f}")
                print(f"  TP: {sl_tp['tp']:.5f}")
                print(f"  Risk/Reward: 1:{sl_tp['risk_reward']:.2f}")

    print(f"\n{'='*80}")
    print(f"Total Signals Found: {len(signals)}")
    print(f"  BUY: {len([s for s in signals if s['signal'] == 'BUY'])}")
    print(f"  SELL: {len([s for s in signals if s['signal'] == 'SELL'])}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    quick_test()
