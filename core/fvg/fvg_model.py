# core/fvg/fvg_model.py
"""
FVG Model - Fair Value Gap Object Definition

Định nghĩa FVG object với đầy đủ thuộc tính và methods
Theo yêu cầu:
- FVG chạm = mất hiệu lực (không có khái niệm lấp 50%)
- Lookback 90 ngày
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal, Optional
import pandas as pd


@dataclass
class FVG:
    """
    Fair Value Gap Object
    
    Attributes:
        fvg_id: Unique identifier
        fvg_type: BULLISH hoặc BEARISH
        created_index: Index trong dataframe khi FVG được tạo
        created_timestamp: Thời gian tạo FVG
        created_candle_indices: Tuple (i-2, i-1, i) của 3 nến tạo FVG
        top: Giá trên của gap
        bottom: Giá dưới của gap
        middle: 50% của gap (auto calculated)
        gap_size: Kích thước gap (auto calculated)
        is_active: FVG còn hiệu lực không
        is_touched: FVG đã bị chạm chưa
        touched_index: Index nến chạm FVG
        touched_timestamp: Thời gian chạm FVG
        touched_price: Giá chạm FVG
        strength: Gap size / ATR (độ mạnh)
        atr_at_creation: ATR tại thời điểm tạo
    """
    
    # ===== BASIC INFO =====
    fvg_id: str
    fvg_type: Literal['BULLISH', 'BEARISH']
    
    # ===== CREATION INFO =====
    created_index: int
    created_timestamp: pd.Timestamp
    created_candle_indices: tuple  # (i-2, i-1, i)
    
    # ===== FVG ZONE =====
    top: float
    bottom: float
    middle: float = field(init=False)
    gap_size: float = field(init=False)
    
    # ===== STATE TRACKING =====
    is_active: bool = True
    is_touched: bool = False
    touched_index: Optional[int] = None
    touched_timestamp: Optional[pd.Timestamp] = None
    touched_price: Optional[float] = None
    
    # ===== METADATA =====
    strength: float = 0.0
    atr_at_creation: float = 0.0
    
    def __post_init__(self):
        """Tính toán các giá trị tự động sau khi init"""
        self.middle = (self.top + self.bottom) / 2
        self.gap_size = abs(self.top - self.bottom)
    
    def check_touched(self, candle_high: float, candle_low: float, 
                     current_index: int, current_timestamp: pd.Timestamp) -> bool:
        """
        Kiểm tra nến hiện tại có CHẠM vào FVG không
        
        ⚠️ QUAN TRỌNG: CHẠM = MẤT HIỆU LỰC ngay lập tức
        
        BULLISH FVG (gap ở dưới):
            - Bị chạm khi candle_low <= self.top
            - Price quay lại xuống chạm vào zone
        
        BEARISH FVG (gap ở trên):
            - Bị chạm khi candle_high >= self.bottom
            - Price quay lại lên chạm vào zone
        
        Args:
            candle_high: High của nến hiện tại
            candle_low: Low của nến hiện tại
            current_index: Index hiện tại
            current_timestamp: Timestamp hiện tại
        
        Returns:
            bool: True nếu bị chạm
        """
        
        if self.is_touched:
            return True  # Đã chạm rồi, không cần check nữa
        
        touched = False
        touch_price = None
        
        if self.fvg_type == 'BULLISH':
            # Bullish FVG ở dưới
            # Bị chạm khi price quay xuống chạm top của gap
            if candle_low <= self.top:
                touched = True
                touch_price = candle_low  # Giá thấp nhất của nến chạm
        
        elif self.fvg_type == 'BEARISH':
            # Bearish FVG ở trên
            # Bị chạm khi price quay lên chạm bottom của gap
            if candle_high >= self.bottom:
                touched = True
                touch_price = candle_high  # Giá cao nhất của nến chạm
        
        # Nếu bị chạm → cập nhật state
        if touched:
            self.is_touched = True
            self.is_active = False  # Mất hiệu lực ngay
            self.touched_index = current_index
            self.touched_timestamp = current_timestamp
            self.touched_price = touch_price
        
        return touched
    
    def is_valid_target(self, current_price: float) -> bool:
        """
        Kiểm tra FVG có phải target hợp lệ không
        
        Điều kiện:
        1. Chưa bị chạm (is_touched = False)
        2. Còn hiệu lực (is_active = True)
        3. Nằm đúng phía so với giá hiện tại
        
        Args:
            current_price: Giá hiện tại
        
        Returns:
            bool: True nếu là target hợp lệ
        """
        
        if not self.is_active or self.is_touched:
            return False
        
        if self.fvg_type == 'BULLISH':
            # Bullish FVG phải ở DƯỚI giá hiện tại
            return current_price > self.top
        
        elif self.fvg_type == 'BEARISH':
            # Bearish FVG phải ở TRÊN giá hiện tại
            return current_price < self.bottom
        
        return False
    
    def get_age_in_candles(self, current_index: int) -> int:
        """
        Tính tuổi của FVG theo số nến
        
        Args:
            current_index: Index hiện tại
        
        Returns:
            int: Số nến từ khi FVG được tạo
        """
        return current_index - self.created_index
    
    def get_age_in_days(self, current_timestamp: pd.Timestamp) -> float:
        """
        Tính tuổi của FVG theo số ngày
        
        Args:
            current_timestamp: Timestamp hiện tại
        
        Returns:
            float: Số ngày từ khi FVG được tạo
        """
        age = current_timestamp - self.created_timestamp
        return age.total_seconds() / (24 * 3600)
    
    def get_distance_to_price(self, current_price: float) -> float:
        """
        Tính khoảng cách từ giá hiện tại đến FVG
        
        Args:
            current_price: Giá hiện tại
        
        Returns:
            float: Khoảng cách (dương = FVG ở xa, âm = price đã vào FVG)
        """
        
        if self.fvg_type == 'BULLISH':
            # Khoảng cách từ price xuống FVG top
            return current_price - self.top
        
        elif self.fvg_type == 'BEARISH':
            # Khoảng cách từ price lên FVG bottom
            return self.bottom - current_price
        
        return 0.0
    
    def to_dict(self) -> dict:
        """
        Convert FVG object thành dictionary để log/export
        
        Returns:
            dict: FVG data
        """
        return {
            'fvg_id': self.fvg_id,
            'fvg_type': self.fvg_type,
            'created_index': self.created_index,
            'created_timestamp': self.created_timestamp,
            'top': self.top,
            'bottom': self.bottom,
            'middle': self.middle,
            'gap_size': self.gap_size,
            'is_active': self.is_active,
            'is_touched': self.is_touched,
            'touched_index': self.touched_index,
            'touched_timestamp': self.touched_timestamp,
            'touched_price': self.touched_price,
            'strength': self.strength,
            'atr_at_creation': self.atr_at_creation
        }
    
    def __repr__(self) -> str:
        """String representation của FVG"""
        status = "ACTIVE" if self.is_active else "TOUCHED"
        return (f"FVG({self.fvg_type}, {self.bottom:.5f}-{self.top:.5f}, "
                f"created={self.created_timestamp}, status={status})")


# ===== HELPER FUNCTIONS =====

def generate_fvg_id(fvg_type: str, timestamp: pd.Timestamp, index: int) -> str:
    """
    Tạo unique ID cho FVG
    
    Args:
        fvg_type: BULLISH hoặc BEARISH
        timestamp: Timestamp tạo FVG
        index: Index tạo FVG
    
    Returns:
        str: Unique ID
    """
    timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    return f"{fvg_type}_{timestamp_str}_{index}"


def calculate_fvg_strength(gap_size: float, atr: float) -> float:
    """
    Tính độ mạnh của FVG
    
    Công thức: strength = gap_size / ATR
    
    Giải thích:
    - strength < 0.5: FVG yếu, có thể bị chạm nhanh
    - strength 0.5-1.0: FVG trung bình
    - strength > 1.0: FVG mạnh, có giá trị cao
    
    Args:
        gap_size: Kích thước gap
        atr: Average True Range
    
    Returns:
        float: Độ mạnh của FVG
    """
    if atr == 0:
        return 0.0
    return gap_size / atr