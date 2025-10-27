"""
Multi-Timeframe Strategy Template

Strategy su dung config tu config.py de tu dong setup cac timeframe.
User chi can thay doi config.MULTI_TIMEFRAME_STRATEGY_CONFIG de adjust strategy.

Author: Claude Code
Date: 2025-10-24
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
import pandas_ta as ta
from core.fvg.multi_timeframe_manager import MultiTimeframeManager
from config import MULTI_TIMEFRAME_STRATEGY_CONFIG, INDICATORS_CONFIG, DATA_CONFIG


class MultiTimeframeStrategy:
    """
    Multi-timeframe strategy using configuration from config.py

    Automatically setup FVG and indicators based on config.
    """

    def __init__(self, data: pd.DataFrame, config: dict = None):
        """
        Initialize strategy

        Args:
            data: Base timeframe data (smallest timeframe)
            config: Strategy config (default: MULTI_TIMEFRAME_STRATEGY_CONFIG)
        """
        self.config = config or MULTI_TIMEFRAME_STRATEGY_CONFIG
        self.base_data = data
        self.base_timeframe = self.config['base_timeframe']

        print(f"\n{'='*80}")
        print("MULTI-TIMEFRAME STRATEGY INITIALIZATION")
        print(f"{'='*80}")
        print(f"Base Timeframe: {self.base_timeframe}")
        print(f"Base Data: {len(data)} candles")
        print(f"Date Range: {data.index[0]} to {data.index[-1]}")

        # Initialize MultiTimeframeManager for FVG
        self._setup_fvg_manager()

        # Calculate indicators on respective timeframes
        self._setup_indicators()

        print(f"{'='*80}")
        print("✅ INITIALIZATION COMPLETE")
        print(f"{'='*80}\n")

    def _setup_fvg_manager(self):
        """Setup FVG manager with configured timeframes"""
        print("\n📊 Setting up FVG Manager...")

        self.mtf_fvg = MultiTimeframeManager(
            self.base_data,
            base_timeframe=self.base_timeframe
        )

        # Add FVG timeframes from config
        fvg_tfs = self.config['fvg_timeframes']

        if fvg_tfs['primary']:
            print(f"  Adding PRIMARY FVG timeframe: {fvg_tfs['primary']}")
            self.mtf_fvg.add_fvg_timeframe(fvg_tfs['primary'])

        if fvg_tfs['secondary']:
            print(f"  Adding SECONDARY FVG timeframe: {fvg_tfs['secondary']}")
            self.mtf_fvg.add_fvg_timeframe(fvg_tfs['secondary'])

        if fvg_tfs['tertiary']:
            print(f"  Adding TERTIARY FVG timeframe: {fvg_tfs['tertiary']}")
            self.mtf_fvg.add_fvg_timeframe(fvg_tfs['tertiary'])

    def _setup_indicators(self):
        """Calculate indicators on configured timeframes"""
        print("\n📈 Calculating Indicators...")

        ind_tfs = self.config['indicator_timeframes']

        # RSI
        if ind_tfs['rsi'] == self.base_timeframe:
            print(f"  RSI on {ind_tfs['rsi']}")
            self.base_data['rsi'] = ta.rsi(
                self.base_data['close'],
                length=INDICATORS_CONFIG['rsi_period']
            )
        else:
            # Need to resample and align (will implement later)
            print(f"  ⚠️  RSI on {ind_tfs['rsi']} (requires resample - TODO)")

        # MACD
        if ind_tfs['macd'] == self.base_timeframe:
            print(f"  MACD on {ind_tfs['macd']}")
            macd = ta.macd(
                self.base_data['close'],
                fast=INDICATORS_CONFIG['macd_fast'],
                slow=INDICATORS_CONFIG['macd_slow'],
                signal=INDICATORS_CONFIG['macd_signal']
            )
            self.base_data['macd'] = macd[f"MACD_{INDICATORS_CONFIG['macd_fast']}_{INDICATORS_CONFIG['macd_slow']}_{INDICATORS_CONFIG['macd_signal']}"]
            self.base_data['macd_signal'] = macd[f"MACDs_{INDICATORS_CONFIG['macd_fast']}_{INDICATORS_CONFIG['macd_slow']}_{INDICATORS_CONFIG['macd_signal']}"]
            self.base_data['macd_hist'] = macd[f"MACDh_{INDICATORS_CONFIG['macd_fast']}_{INDICATORS_CONFIG['macd_slow']}_{INDICATORS_CONFIG['macd_signal']}"]

        # ATR
        if ind_tfs['atr'] == self.base_timeframe:
            print(f"  ATR on {ind_tfs['atr']}")
            self.base_data['atr'] = ta.atr(
                self.base_data['high'],
                self.base_data['low'],
                self.base_data['close'],
                length=INDICATORS_CONFIG['atr_period']
            )

        # Volume SMA
        if ind_tfs['volume_sma'] == self.base_timeframe:
            print(f"  Volume SMA on {ind_tfs['volume_sma']}")
            self.base_data['volume_sma'] = ta.sma(
                self.base_data['volume'],
                length=20
            )

    def analyze(self, index: int) -> dict:
        """
        Analyze market at given index

        Args:
            index: Current index in base timeframe

        Returns:
            dict: Analysis result with signals
        """
        # Update FVG manager
        self.mtf_fvg.update(index)

        # Get FVG bias from configured timeframes
        fvg_tfs = self.config['fvg_timeframes']
        fvg_analysis = {}

        if fvg_tfs['primary']:
            fvg_analysis['primary'] = self.mtf_fvg.get_fvg_structure(
                fvg_tfs['primary'], index
            )

        if fvg_tfs['secondary']:
            fvg_analysis['secondary'] = self.mtf_fvg.get_fvg_structure(
                fvg_tfs['secondary'], index
            )

        if fvg_tfs['tertiary']:
            fvg_analysis['tertiary'] = self.mtf_fvg.get_fvg_structure(
                fvg_tfs['tertiary'], index
            )

        # Get indicator values
        indicators = {}
        if 'rsi' in self.base_data.columns:
            indicators['rsi'] = self.base_data.iloc[index]['rsi']

        if 'macd' in self.base_data.columns:
            indicators['macd'] = self.base_data.iloc[index]['macd']
            indicators['macd_signal'] = self.base_data.iloc[index]['macd_signal']
            indicators['macd_hist'] = self.base_data.iloc[index]['macd_hist']

        if 'atr' in self.base_data.columns:
            indicators['atr'] = self.base_data.iloc[index]['atr']

        if 'volume_sma' in self.base_data.columns:
            volume = self.base_data.iloc[index]['volume']
            volume_sma = self.base_data.iloc[index]['volume_sma']
            indicators['volume_ratio'] = volume / volume_sma if volume_sma > 0 else 0

        # Generate signal
        signal = self._generate_signal(fvg_analysis, indicators)

        return {
            'fvg': fvg_analysis,
            'indicators': indicators,
            'signal': signal,
            'price': self.base_data.iloc[index]['close'],
            'timestamp': self.base_data.index[index]
        }

    def _generate_signal(self, fvg_analysis: dict, indicators: dict) -> str:
        """
        Generate trading signal based on FVG and indicators

        Args:
            fvg_analysis: FVG analysis from multiple timeframes
            indicators: Indicator values

        Returns:
            str: 'BUY', 'SELL', or 'NEUTRAL'
        """
        # Get primary FVG bias
        primary_fvg = fvg_analysis.get('primary')
        if not primary_fvg:
            return 'NEUTRAL'

        primary_bias = primary_fvg.get('bias')

        # BUY conditions
        if primary_bias == 'BULLISH_BIAS':
            # Check RSI oversold
            if indicators.get('rsi', 50) < INDICATORS_CONFIG['rsi_oversold']:
                # Check volume spike
                if indicators.get('volume_ratio', 0) > 1.5:
                    return 'BUY'

        # SELL conditions
        elif primary_bias == 'BEARISH_BIAS':
            # Check RSI overbought
            if indicators.get('rsi', 50) > INDICATORS_CONFIG['rsi_overbought']:
                # Check volume spike
                if indicators.get('volume_ratio', 0) > 1.5:
                    return 'SELL'

        return 'NEUTRAL'

    def backtest(self, start_index: int = 100) -> list:
        """
        Run backtest from start_index to end

        Args:
            start_index: Starting index

        Returns:
            list: List of signals
        """
        print(f"\n{'='*80}")
        print("RUNNING BACKTEST")
        print(f"{'='*80}")
        print(f"Start Index: {start_index}")
        print(f"End Index: {len(self.base_data) - 1}")
        print(f"Total Candles: {len(self.base_data) - start_index}")

        signals = []

        for i in range(start_index, len(self.base_data)):
            analysis = self.analyze(i)

            if analysis['signal'] != 'NEUTRAL':
                signals.append(analysis)

                # Print signal
                print(f"\n{analysis['signal']} Signal at {analysis['timestamp']}")
                print(f"  Price: {analysis['price']:.5f}")

                # FVG info
                if analysis['fvg'].get('primary'):
                    fvg = analysis['fvg']['primary']
                    print(f"  FVG Bias: {fvg['bias']}")
                    print(f"  Active FVGs: {fvg['total_active_fvgs']}")

                # Indicators
                ind = analysis['indicators']
                if 'rsi' in ind:
                    print(f"  RSI: {ind['rsi']:.2f}")
                if 'volume_ratio' in ind:
                    print(f"  Volume Ratio: {ind['volume_ratio']:.2f}x")

        print(f"\n{'='*80}")
        print("BACKTEST COMPLETE")
        print(f"{'='*80}")
        print(f"Total Signals: {len(signals)}")
        print(f"  BUY: {len([s for s in signals if s['signal'] == 'BUY'])}")
        print(f"  SELL: {len([s for s in signals if s['signal'] == 'SELL'])}")

        return signals


def main():
    """Main function to run strategy"""

    # Load data
    print("Loading data...")
    data_path = f"data/{DATA_CONFIG['symbol']}_{DATA_CONFIG['timeframe']}_{DATA_CONFIG['days']}days.csv"
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"Loaded {len(data)} candles from {data_path}")

    # Initialize strategy
    strategy = MultiTimeframeStrategy(data)

    # Run backtest
    signals = strategy.backtest(start_index=100)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Configuration:")
    print(f"  Base TF: {MULTI_TIMEFRAME_STRATEGY_CONFIG['base_timeframe']}")
    print(f"  Primary FVG TF: {MULTI_TIMEFRAME_STRATEGY_CONFIG['fvg_timeframes']['primary']}")
    print(f"  RSI TF: {MULTI_TIMEFRAME_STRATEGY_CONFIG['indicator_timeframes']['rsi']}")
    print(f"  MACD TF: {MULTI_TIMEFRAME_STRATEGY_CONFIG['indicator_timeframes']['macd']}")
    print(f"\nResults:")
    print(f"  Total Signals: {len(signals)}")
    print("="*80)


if __name__ == '__main__':
    main()
