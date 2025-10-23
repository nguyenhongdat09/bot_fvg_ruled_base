# core/fvg/fvg_detector.py
"""
FVG Detector - Phát hiện Fair Value Gap mới

Nhiệm vụ:
- Scan 3 nến liên tiếp để tìm gap
- Validate gap (đủ lớn, không phải noise)
- Tạo FVG object
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .fvg_model import FVG, generate_fvg_id, calculate_fvg_strength


class FVGDetector:
    """
    FVG Detector - Phát hiện FVG mới trong dữ liệu
    
    Attributes:
        min_gap_atr_ratio: Gap tối thiểu = ATR × ratio (mặc định 0.3)
                          Lọc bỏ gap quá nhỏ (noise)
        min_gap_pips: Gap tối thiểu theo pips (optional, mặc định None)
    """
    
    def __init__(self, min_gap_atr_ratio: float = 0.3, min_gap_pips: Optional[float] = None):
        """
        Initialize FVG Detector
        
        Args:
            min_gap_atr_ratio: Gap phải >= ATR × ratio
            min_gap_pips: Gap phải >= pips (optional)
        """
        self.min_gap_atr_ratio = min_gap_atr_ratio
        self.min_gap_pips = min_gap_pips
    
    def detect_fvg_at_index(self, data: pd.DataFrame, index: int, atr: float) -> Optional[FVG]:
        """
        Phát hiện FVG tại một index cụ thể
        
        FVG được tạo bởi 3 nến liên tiếp:
        - Nến i-2: First candle
        - Nến i-1: Middle candle (tạo gap)
        - Nến i: Third candle (confirm gap)
        
        BULLISH FVG: High[i-2] < Low[i] (có gap ở giữa)
        BEARISH FVG: Low[i-2] > High[i] (có gap ở giữa)
        
        Args:
            data: DataFrame chứa OHLC data
            index: Index cần check
            atr: ATR value tại thời điểm này
        
        Returns:
            FVG object nếu phát hiện, None nếu không
        """
        
        # Cần ít nhất 3 nến
        if index < 2:
            return None
        
        # Lấy 3 nến
        candle_i = data.iloc[index]       # Nến hiện tại
        candle_i1 = data.iloc[index - 1]  # Nến trước 1
        candle_i2 = data.iloc[index - 2]  # Nến trước 2
        
        fvg = None
        
        # ===== CHECK BULLISH FVG =====
        if candle_i2['high'] < candle_i['low']:
            gap_size = candle_i['low'] - candle_i2['high']
            
            # Validate gap size
            if self._is_valid_gap(gap_size, atr):
                fvg_id = generate_fvg_id('BULLISH', candle_i.name, index)
                strength = calculate_fvg_strength(gap_size, atr)
                
                fvg = FVG(
                    fvg_id=fvg_id,
                    fvg_type='BULLISH',
                    created_index=index,
                    created_timestamp=candle_i.name,
                    created_candle_indices=(index - 2, index - 1, index),
                    top=candle_i['low'],
                    bottom=candle_i2['high'],
                    strength=strength,
                    atr_at_creation=atr
                )
        
        # ===== CHECK BEARISH FVG =====
        elif candle_i2['low'] > candle_i['high']:
            gap_size = candle_i2['low'] - candle_i['high']
            
            # Validate gap size
            if self._is_valid_gap(gap_size, atr):
                fvg_id = generate_fvg_id('BEARISH', candle_i.name, index)
                strength = calculate_fvg_strength(gap_size, atr)
                
                fvg = FVG(
                    fvg_id=fvg_id,
                    fvg_type='BEARISH',
                    created_index=index,
                    created_timestamp=candle_i.name,
                    created_candle_indices=(index - 2, index - 1, index),
                    top=candle_i2['low'],
                    bottom=candle_i['high'],
                    strength=strength,
                    atr_at_creation=atr
                )
        
        return fvg
    
    def detect_all_fvgs(self, data: pd.DataFrame, atr_series: pd.Series, 
                       start_index: int = 2) -> List[FVG]:
        """
        Phát hiện tất cả FVG trong một đoạn data
        
        Args:
            data: DataFrame chứa OHLC
            atr_series: Series chứa ATR values
            start_index: Index bắt đầu scan (mặc định 2)
        
        Returns:
            List[FVG]: Danh sách FVG phát hiện được
        """
        
        fvgs = []
        
        for i in range(start_index, len(data)):
            atr = atr_series.iloc[i]
            fvg = self.detect_fvg_at_index(data, i, atr)
            
            if fvg is not None:
                fvgs.append(fvg)
        
        return fvgs
    
    def _is_valid_gap(self, gap_size: float, atr: float) -> bool:
        """
        Kiểm tra gap có đủ lớn không (lọc noise)
        
        Điều kiện:
        1. Gap >= ATR × min_gap_atr_ratio
        2. (Optional) Gap >= min_gap_pips
        
        Args:
            gap_size: Kích thước gap
            atr: ATR value
        
        Returns:
            bool: True nếu gap hợp lệ
        """
        
        # Check ATR ratio
        if gap_size < atr * self.min_gap_atr_ratio:
            return False
        
        # Check min pips (optional)
        if self.min_gap_pips is not None:
            # Convert pips to price (assuming 5-digit forex)
            min_gap_price = self.min_gap_pips * 0.0001
            if gap_size < min_gap_price:
                return False
        
        return True
    
    def get_statistics(self, fvgs: List[FVG]) -> dict:
        """
        Thống kê FVG đã phát hiện
        
        Args:
            fvgs: Danh sách FVG
        
        Returns:
            dict: Thống kê
        """
        
        if not fvgs:
            return {
                'total': 0,
                'bullish': 0,
                'bearish': 0,
                'avg_gap_size': 0,
                'avg_strength': 0
            }
        
        bullish_count = sum(1 for fvg in fvgs if fvg.fvg_type == 'BULLISH')
        bearish_count = sum(1 for fvg in fvgs if fvg.fvg_type == 'BEARISH')
        
        avg_gap_size = np.mean([fvg.gap_size for fvg in fvgs])
        avg_strength = np.mean([fvg.strength for fvg in fvgs])
        
        return {
            'total': len(fvgs),
            'bullish': bullish_count,
            'bearish': bearish_count,
            'avg_gap_size': avg_gap_size,
            'avg_strength': avg_strength
        }