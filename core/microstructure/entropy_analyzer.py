# core/microstructure/entropy_analyzer.py
"""
Entropy-Based Microstructure Analysis - Đo "chaos" và predictability

Shannon Entropy trong trading:
- Low entropy = Ordered, predictable, trending
- High entropy = Chaotic, unpredictable, ranging/exhausted

Applications:
1. Detect consolidation vs trending
2. Measure market efficiency
3. Identify regime changes

Use Case:
Price di xa FVG → Entropy tăng → Market chaos/exhaustion → Reversal likely
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class EntropyAnalyzer:
    """
    Entropy-Based Market Analysis

    Attributes:
        entropy_window: Window for entropy calculation
        n_bins: Number of bins for discretization
        permutation_order: Order for permutation entropy (3-7)
    """

    def __init__(self, entropy_window: int = 20,
                 n_bins: int = 10,
                 permutation_order: int = 3):
        """
        Initialize Entropy Analyzer

        Args:
            entropy_window: Rolling window for entropy calculation
            n_bins: Number of bins for discretization
            permutation_order: Order for permutation entropy
        """
        self.entropy_window = entropy_window
        self.n_bins = n_bins
        self.permutation_order = permutation_order

    def calculate_shannon_entropy(self, data: pd.Series) -> float:
        """
        Calculate Shannon Entropy of a series

        H = -Σ(p(x) * log2(p(x)))

        Where p(x) is probability of value x

        Args:
            data: Data series

        Returns:
            float: Entropy (0 to log2(n_bins))
        """
        if len(data) < 2:
            return 0.0

        # Discretize data into bins
        counts, _ = np.histogram(data, bins=self.n_bins)

        # Calculate probabilities
        probabilities = counts / len(data)

        # Remove zeros (log(0) is undefined)
        probabilities = probabilities[probabilities > 0]

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy

    def calculate_rolling_entropy(self, data: pd.DataFrame,
                                  price_col: str = 'close') -> pd.Series:
        """
        Calculate rolling Shannon entropy of returns

        Args:
            data: DataFrame with price data
            price_col: Column to use for calculation

        Returns:
            pd.Series: Rolling entropy values
        """
        # Calculate returns
        returns = data[price_col].pct_change().fillna(0)

        # Rolling entropy
        entropy_values = returns.rolling(
            window=self.entropy_window,
            min_periods=self.entropy_window // 2
        ).apply(self.calculate_shannon_entropy, raw=True)

        return entropy_values

    def calculate_permutation_entropy(self, data: np.ndarray, order: int = 3) -> float:
        """
        Calculate Permutation Entropy

        More robust than Shannon entropy for time series
        Captures temporal patterns

        Args:
            data: Time series data
            order: Embedding dimension (typically 3-7)

        Returns:
            float: Permutation entropy (0-1)
        """
        if len(data) < order:
            return 0.5  # Neutral

        # Create permutation patterns
        permutations = {}

        for i in range(len(data) - order + 1):
            window = data[i:i+order]
            # Get ranking of values in window
            ranking = tuple(stats.rankdata(window, method='ordinal') - 1)
            permutations[ranking] = permutations.get(ranking, 0) + 1

        # Calculate probabilities
        n_patterns = len(data) - order + 1
        probabilities = np.array(list(permutations.values())) / n_patterns

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

        # Normalize to 0-1
        max_entropy = np.log2(np.math.factorial(order))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return normalized_entropy

    def calculate_rolling_permutation_entropy(self, data: pd.DataFrame,
                                             price_col: str = 'close') -> pd.Series:
        """
        Calculate rolling permutation entropy

        Args:
            data: DataFrame with price data
            price_col: Column to use

        Returns:
            pd.Series: Rolling permutation entropy (0-1)
        """
        prices = data[price_col].values

        entropy_values = []

        for i in range(len(prices)):
            if i < self.entropy_window:
                entropy_values.append(np.nan)
                continue

            window = prices[i-self.entropy_window:i]
            perm_entropy = self.calculate_permutation_entropy(
                window, order=self.permutation_order
            )
            entropy_values.append(perm_entropy)

        return pd.Series(entropy_values, index=data.index)

    def calculate_approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Calculate Approximate Entropy (ApEn)

        Measures regularity/predictability in time series
        Lower ApEn = more regular, more predictable

        Args:
            data: Time series
            m: Pattern length
            r: Tolerance (fraction of std)

        Returns:
            float: ApEn value
        """
        if len(data) < m + 1:
            return 0.5

        # Normalize data
        data_std = np.std(data)
        if data_std == 0:
            return 0.0

        r_threshold = r * data_std

        def _maxdist(xi, xj):
            """Maximum distance between patterns"""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m_val):
            """Calculate phi(m)"""
            patterns = np.array([[data[j] for j in range(i, i + m_val)]
                                for i in range(len(data) - m_val + 1)])

            C = []
            for i in range(len(patterns)):
                count = sum([1 for j in range(len(patterns))
                           if _maxdist(patterns[i], patterns[j]) <= r_threshold])
                C.append(count / len(patterns))

            return np.mean([np.log(c) for c in C if c > 0])

        return abs(_phi(m + 1) - _phi(m))

    def calculate_rolling_approximate_entropy(self, data: pd.DataFrame,
                                             price_col: str = 'close') -> pd.Series:
        """
        Calculate rolling approximate entropy

        Args:
            data: DataFrame with price data
            price_col: Column to use

        Returns:
            pd.Series: Rolling ApEn values
        """
        prices = data[price_col].values

        apen_values = []

        for i in range(len(prices)):
            if i < self.entropy_window:
                apen_values.append(np.nan)
                continue

            window = prices[i-self.entropy_window:i]
            apen = self.calculate_approximate_entropy(window)
            apen_values.append(apen)

        return pd.Series(apen_values, index=data.index)

    def get_entropy_regime(self, entropy_value: float,
                          entropy_type: str = 'permutation') -> str:
        """
        Classify market regime based on entropy

        Args:
            entropy_value: Entropy value
            entropy_type: 'shannon', 'permutation', or 'approximate'

        Returns:
            str: Market regime
        """
        if entropy_type == 'permutation' or entropy_type == 'approximate':
            # Permutation/ApEn entropy: 0-1
            if entropy_value < 0.3:
                return 'HIGHLY_ORDERED'  # Strong trend
            elif entropy_value < 0.5:
                return 'ORDERED'  # Trending
            elif entropy_value < 0.7:
                return 'TRANSITIONAL'  # Mixed
            else:
                return 'CHAOTIC'  # Ranging/exhausted
        else:
            # Shannon entropy: depends on n_bins
            max_entropy = np.log2(self.n_bins)
            normalized = entropy_value / max_entropy

            if normalized < 0.3:
                return 'HIGHLY_ORDERED'
            elif normalized < 0.5:
                return 'ORDERED'
            elif normalized < 0.7:
                return 'TRANSITIONAL'
            else:
                return 'CHAOTIC'

    def analyze_entropy(self, data: pd.DataFrame, index: int) -> Dict:
        """
        Complete entropy analysis at given index

        Args:
            data: DataFrame with OHLCV
            index: Current index

        Returns:
            dict: {
                'shannon_entropy': float,
                'permutation_entropy': float,
                'approximate_entropy': float,
                'regime': str,
                'is_chaotic': bool,
                'is_ordered': bool,
                'entropy_score': float (0-1, high = chaotic)
            }
        """
        if index < self.entropy_window:
            return {
                'shannon_entropy': np.nan,
                'permutation_entropy': np.nan,
                'approximate_entropy': np.nan,
                'regime': 'UNKNOWN',
                'is_chaotic': False,
                'is_ordered': False,
                'entropy_score': 0.5
            }

        # Calculate all entropy measures
        shannon_ent = self.calculate_rolling_entropy(data.iloc[:index+1])
        perm_ent = self.calculate_rolling_permutation_entropy(data.iloc[:index+1])
        apen = self.calculate_rolling_approximate_entropy(data.iloc[:index+1])

        # Get current values
        shannon_val = shannon_ent.iloc[-1] if not pd.isna(shannon_ent.iloc[-1]) else 0.5
        perm_val = perm_ent.iloc[-1] if not pd.isna(perm_ent.iloc[-1]) else 0.5
        apen_val = apen.iloc[-1] if not pd.isna(apen.iloc[-1]) else 0.5

        # Average entropy score (use permutation as primary)
        entropy_score = perm_val

        # Determine regime
        regime = self.get_entropy_regime(perm_val, 'permutation')

        return {
            'shannon_entropy': shannon_val,
            'permutation_entropy': perm_val,
            'approximate_entropy': apen_val,
            'regime': regime,
            'is_chaotic': regime in ['CHAOTIC', 'TRANSITIONAL'],
            'is_ordered': regime in ['HIGHLY_ORDERED', 'ORDERED'],
            'entropy_score': entropy_score
        }

    def analyze_entropy_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze entropy for entire series

        Args:
            data: DataFrame with OHLCV

        Returns:
            DataFrame with entropy columns added
        """
        result = data.copy()

        # Calculate all entropy measures
        result['shannon_entropy'] = self.calculate_rolling_entropy(data)
        result['permutation_entropy'] = self.calculate_rolling_permutation_entropy(data)
        result['approximate_entropy'] = self.calculate_rolling_approximate_entropy(data)

        # Determine regimes
        result['entropy_regime'] = result['permutation_entropy'].apply(
            lambda x: self.get_entropy_regime(x, 'permutation') if not pd.isna(x) else 'UNKNOWN'
        )

        result['is_chaotic'] = result['entropy_regime'].isin(['CHAOTIC', 'TRANSITIONAL'])
        result['is_ordered'] = result['entropy_regime'].isin(['HIGHLY_ORDERED', 'ORDERED'])

        return result

    def detect_entropy_shifts(self, data: pd.DataFrame,
                             threshold: float = 0.2) -> pd.Series:
        """
        Detect significant shifts in entropy (regime changes)

        Args:
            data: DataFrame with OHLCV
            threshold: Threshold for significant shift

        Returns:
            pd.Series: Shift signals (1=order->chaos, -1=chaos->order, 0=no shift)
        """
        perm_ent = self.calculate_rolling_permutation_entropy(data)

        # Calculate changes in entropy
        entropy_change = perm_ent.diff(5)

        shifts = pd.Series(0, index=data.index)

        # Order -> Chaos (entropy increasing)
        shifts[entropy_change > threshold] = 1

        # Chaos -> Order (entropy decreasing)
        shifts[entropy_change < -threshold] = -1

        return shifts
