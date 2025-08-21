"""
Stock Pattern Scanner - Improved Version
Advanced stock pattern detection with traditional algorithms and lightweight ML
Key improvements:
- Fixed ML feature extraction and model training
- Better error handling and data validation
- Improved UI with caching and performance optimizations
- Enhanced pattern detection algorithms
- Better visualization and user experience
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# =========================== CONFIGURATION ===========================

@dataclass
class PatternConfig:
    """Configuration for pattern detection parameters"""
    shoulder_tolerance: float = 0.08
    neckline_tolerance: float = 0.05
    min_duration_days: int = 30
    breakout_threshold: float = 0.02
    price_tolerance: float = 0.03
    min_rebound: float = 0.05
    rim_tolerance: float = 0.05
    min_cup_depth: float = 0.12
    handle_retrace_min: float = 0.2
    handle_retrace_max: float = 0.5

DEFAULT_SCAN_DAYS = 365
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
PATTERN_CONFIG = PatternConfig()

PATTERN_DISPLAY = {
    'Head and Shoulders': {'icon': '📉', 'color': '#ff6b6b', 'type': 'bearish'},
    'Double Bottom': {'icon': '📈', 'color': '#51cf66', 'type': 'bullish'},
    'Double Top': {'icon': '📉', 'color': '#ff8787', 'type': 'bearish'},
    'Cup and Handle': {'icon': '☕', 'color': '#74c0fc', 'type': 'bullish'},
    'Triangle': {'icon': '📐', 'color': '#ffd43b', 'type': 'neutral'}
}

# =========================== DATA HANDLING ===========================

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def fetch_stock_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch stock data with caching and error handling"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
        
        if df.empty or len(df) < 30:
            return None
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure numeric columns and handle missing data
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Forward fill missing values
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # Remove any remaining NaN rows
        df = df.dropna(subset=['Close'])
        
        if len(df) < 30:
            return None
            
        return df
        
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {str(e)}")
        return None

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate DataFrame structure and data quality"""
    if df is None or df.empty:
        return False
    
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return False
    
    if len(df) < 30:
        return False
    
    # Check for reasonable price data
    if df['Close'].isnull().sum() > len(df) * 0.1:  # More than 10% missing
        return False
        
    return True

# =========================== IMPROVED PATTERN DETECTION ===========================

class PatternDetector:
    """Improved pattern detection with better algorithms"""
    
    def __init__(self, config: PatternConfig = PATTERN_CONFIG):
        self.config = config
        
    def find_peaks_troughs(self, prices: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find peaks and troughs with improved detection"""
        # Use different orders for different price ranges
        adaptive_order = max(3, min(order, len(prices) // 10))
        
        peaks = argrelextrema(prices, np.greater, order=adaptive_order)[0]
        troughs = argrelextrema(prices, np.less, order=adaptive_order)[0]
        
        # Filter out insignificant peaks/troughs
        if len(peaks) > 0:
            peak_heights = prices[peaks]
            significant_peaks = peaks[peak_heights > np.percentile(prices, 60)]
            peaks = significant_peaks
            
        if len(troughs) > 0:
            trough_depths = prices[troughs]
            significant_troughs = troughs[trough_depths < np.percentile(prices, 40)]
            troughs = significant_troughs
            
        return peaks, troughs
    
    def detect_head_and_shoulders(self, df: pd.DataFrame) -> List[Dict]:
        """Enhanced Head and Shoulders detection with advanced technical analysis"""
        if not validate_dataframe(df) or len(df) < 60:
            return []
        
        prices = df['Close'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        dates = df['Date'] if 'Date' in df.columns else df.index
        peaks, troughs = self.find_peaks_troughs(prices)
        
        if len(peaks) < 3 or len(troughs) < 2:
            return []
        
        patterns = []
        
        for i in range(len(peaks) - 2):
            for j in range(i + 1, len(peaks) - 1):
                for k in range(j + 1, len(peaks)):
                    ls_idx, head_idx, rs_idx = peaks[i], peaks[j], peaks[k]
                    ls_price, head_price, rs_price = prices[ls_idx], prices[head_idx], prices[rs_idx]
                    
                    # Enhanced head and shoulders validation
                    # 1. Head should be significantly higher than shoulders (minimum 2% difference)
                    if not (head_price > ls_price * 1.02 and head_price > rs_price * 1.02):
                        continue
                    
                    # 2. Shoulders should be similar height (within tolerance)
                    shoulder_diff = abs(ls_price - rs_price) / min(ls_price, rs_price)
                    if shoulder_diff > self.config.shoulder_tolerance:
                        continue
                    
                    # 3. Find neckline troughs (support levels)
                    left_troughs = [t for t in troughs if ls_idx < t < head_idx]
                    right_troughs = [t for t in troughs if head_idx < t < rs_idx]
                    
                    if not left_troughs or not right_troughs:
                        continue
                    
                    left_trough_idx = min(left_troughs, key=lambda x: prices[x])
                    right_trough_idx = min(right_troughs, key=lambda x: prices[x])
                    
                    neckline_left = prices[left_trough_idx]
                    neckline_right = prices[right_trough_idx]
                    neckline_level = (neckline_left + neckline_right) / 2
                    
                    # 4. Neckline should be relatively flat (within tolerance)
                    neckline_diff = abs(neckline_left - neckline_right) / neckline_level
                    if neckline_diff > self.config.neckline_tolerance:
                        continue
                    
                    # 5. Pattern duration and timing analysis
                    duration = (dates.iloc[rs_idx] - dates.iloc[ls_idx]).days
                    if duration < self.config.min_duration_days or duration > 200:
                        continue
                    
                    # 6. Volume analysis for confirmation
                    volume_confirmation = True
                    volume_pattern_score = 1.0
                    if volumes is not None:
                        volume_confirmation, volume_pattern_score = self._analyze_hs_volume_pattern(
                            volumes, ls_idx, head_idx, rs_idx, left_trough_idx, right_trough_idx
                        )
                    
                    # 7. Pattern symmetry analysis
                    symmetry_score = self._analyze_hs_symmetry(prices, ls_idx, head_idx, rs_idx)
                    
                    # 8. Support/resistance level validation
                    support_level_score = self._validate_hs_support_level(prices, left_trough_idx, right_trough_idx)
                    
                    # 9. Calculate enhanced confidence
                    confidence = self._calculate_enhanced_hs_confidence(
                        shoulder_diff, neckline_diff, head_price, neckline_level, duration,
                        volume_pattern_score, symmetry_score, support_level_score, volume_confirmation
                    )
                    
                    if confidence < 0.4:  # Lower threshold for more patterns
                        continue
                    
                    # 10. Check for breakout with volume confirmation
                    breakout_idx = self._check_enhanced_breakout(
                        df, right_trough_idx, neckline_level, 'down'
                    )
                    
                    # 11. Calculate target price using measured move
                    pattern_height = head_price - neckline_level
                    target_price = neckline_level - pattern_height
                    
                    pattern = {
                        'type': 'Head and Shoulders',
                        'left_shoulder_idx': int(ls_idx),
                        'head_idx': int(head_idx),
                        'right_shoulder_idx': int(rs_idx),
                        'left_trough_idx': int(left_trough_idx),
                        'right_trough_idx': int(right_trough_idx),
                        'breakout_idx': breakout_idx,
                        'left_shoulder_price': float(ls_price),
                        'head_price': float(head_price),
                        'right_shoulder_price': float(rs_price),
                        'neckline_level': float(neckline_level),
                        'target_price': float(target_price),
                        'pattern_height': float(pattern_height),
                        'confidence': confidence,
                        'status': 'confirmed' if breakout_idx else 'forming',
                        'duration_days': duration,
                        'symmetry_score': symmetry_score,
                        'volume_confirmation': volume_confirmation,
                        'detection_method': 'traditional_enhanced'
                    }
                    
                    patterns.append(pattern)
        
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def detect_double_bottom(self, df: pd.DataFrame) -> List[Dict]:
        """Enhanced Double Bottom detection with advanced technical analysis"""
        if not validate_dataframe(df) or len(df) < 40:
            return []
        
        prices = df['Close'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        dates = df['Date'] if 'Date' in df.columns else df.index
        peaks, troughs = self.find_peaks_troughs(prices)
        
        if len(troughs) < 2 or len(peaks) < 1:
            return []
        
        patterns = []
        
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                t1_idx, t2_idx = troughs[i], troughs[j]
                t1_price, t2_price = prices[t1_idx], prices[t2_idx]
                
                # Enhanced trough similarity check with volume confirmation
                price_diff = abs(t1_price - t2_price) / min(t1_price, t2_price)
                if price_diff > self.config.price_tolerance:
                    continue
                
                # Volume analysis for trough validation
                volume_confirmation = True
                if volumes is not None:
                    # Check if volume is higher at second trough (bullish signal)
                    t1_volume = volumes[t1_idx] if t1_idx < len(volumes) else 0
                    t2_volume = volumes[t2_idx] if t2_idx < len(volumes) else 0
                    volume_confirmation = t2_volume > t1_volume * 0.8  # Allow some flexibility
                
                # Find peak between troughs (neckline)
                between_peaks = [p for p in peaks if t1_idx < p < t2_idx]
                if not between_peaks:
                    continue
                
                neckline_idx = max(between_peaks, key=lambda x: prices[x])
                neckline_price = prices[neckline_idx]
                
                # Enhanced rebound strength analysis
                min_trough = min(t1_price, t2_price)
                rebound = (neckline_price - min_trough) / min_trough
                if rebound < self.config.min_rebound:
                    continue
                
                # Pattern duration and symmetry analysis
                duration = (dates.iloc[t2_idx] - dates.iloc[t1_idx]).days
                if duration < self.config.min_duration_days or duration > 200:  # Reasonable range
                    continue
                
                # Check for W-shape formation
                w_shape_score = self._analyze_w_shape(prices, t1_idx, t2_idx, neckline_idx)
                
                # Volume pattern analysis
                volume_pattern_score = 1.0
                if volumes is not None:
                    volume_pattern_score = self._analyze_volume_pattern(volumes, t1_idx, t2_idx, neckline_idx)
                
                # Support level validation
                support_level_score = self._validate_support_level(prices, t1_idx, t2_idx)
                
                # Calculate enhanced confidence
                confidence = self._calculate_enhanced_db_confidence(
                    price_diff, rebound, duration, w_shape_score, 
                    volume_pattern_score, support_level_score, volume_confirmation
                )
                
                if confidence < 0.4:  # Lower threshold for more patterns
                    continue
                
                # Check for breakout with volume confirmation
                breakout_idx = self._check_enhanced_breakout(df, t2_idx, neckline_price, 'up')
                
                # Calculate target price using measured move
                pattern_height = neckline_price - min_trough
                target_price = neckline_price + pattern_height
                
                pattern = {
                    'type': 'Double Bottom',
                    'first_trough_idx': int(t1_idx),
                    'second_trough_idx': int(t2_idx),
                    'neckline_idx': int(neckline_idx),
                    'breakout_idx': breakout_idx,
                    'first_trough_price': float(t1_price),
                    'second_trough_price': float(t2_price),
                    'neckline_price': float(neckline_price),
                    'target_price': float(target_price),
                    'pattern_height': float(pattern_height),
                    'confidence': confidence,
                    'status': 'confirmed' if breakout_idx else 'forming',
                    'duration_days': duration,
                    'rebound_strength': rebound,
                    'w_shape_score': w_shape_score,
                    'volume_confirmation': volume_confirmation,
                    'detection_method': 'traditional_enhanced'
                }
                
                patterns.append(pattern)
        
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def detect_double_top(self, df: pd.DataFrame) -> List[Dict]:
        """Enhanced Double Top detection with advanced technical analysis"""
        if not validate_dataframe(df) or len(df) < 40:
            return []
        
        prices = df['Close'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        dates = df['Date'] if 'Date' in df.columns else df.index
        peaks, troughs = self.find_peaks_troughs(prices)
        
        if len(peaks) < 2 or len(troughs) < 1:
            return []
        
        patterns = []
        
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                p1_idx, p2_idx = peaks[i], peaks[j]
                p1_price, p2_price = prices[p1_idx], prices[p2_idx]
                
                # Enhanced peak similarity check with volume confirmation
                price_diff = abs(p1_price - p2_price) / min(p1_price, p2_price)
                if price_diff > self.config.price_tolerance:
                    continue
                
                # Volume analysis for peak validation
                volume_confirmation = True
                if volumes is not None:
                    # Check if volume is lower at second peak (bearish signal)
                    p1_volume = volumes[p1_idx] if p1_idx < len(volumes) else 0
                    p2_volume = volumes[p2_idx] if p2_idx < len(volumes) else 0
                    volume_confirmation = p2_volume < p1_volume * 1.2  # Allow some flexibility
                
                # Find trough between peaks (neckline)
                between_troughs = [t for t in troughs if p1_idx < t < p2_idx]
                if not between_troughs:
                    continue
                
                neckline_idx = min(between_troughs, key=lambda x: prices[x])
                neckline_price = prices[neckline_idx]
                
                # Enhanced pullback strength analysis
                max_peak = max(p1_price, p2_price)
                pullback = (max_peak - neckline_price) / max_peak
                if pullback < self.config.min_rebound:
                    continue
                
                # Pattern duration and symmetry analysis
                duration = (dates.iloc[p2_idx] - dates.iloc[p1_idx]).days
                if duration < self.config.min_duration_days or duration > 200:  # Reasonable range
                    continue
                
                # Check for M-shape formation
                m_shape_score = self._analyze_m_shape(prices, p1_idx, p2_idx, neckline_idx)
                
                # Volume pattern analysis
                volume_pattern_score = 1.0
                if volumes is not None:
                    volume_pattern_score = self._analyze_volume_pattern_bearish(volumes, p1_idx, p2_idx, neckline_idx)
                
                # Resistance level validation
                resistance_level_score = self._validate_resistance_level(prices, p1_idx, p2_idx)
                
                # Calculate enhanced confidence
                confidence = self._calculate_enhanced_dt_confidence(
                    price_diff, pullback, duration, m_shape_score, 
                    volume_pattern_score, resistance_level_score, volume_confirmation
                )
                
                if confidence < 0.4:  # Lower threshold for more patterns
                    continue
                
                # Check for breakdown with volume confirmation
                breakdown_idx = self._check_enhanced_breakout(df, p2_idx, neckline_price, 'down')
                
                # Calculate target price using measured move
                pattern_height = max_peak - neckline_price
                target_price = neckline_price - pattern_height
                
                pattern = {
                    'type': 'Double Top',
                    'first_peak_idx': int(p1_idx),
                    'second_peak_idx': int(p2_idx),
                    'neckline_idx': int(neckline_idx),
                    'breakdown_idx': breakdown_idx,
                    'first_peak_price': float(p1_price),
                    'second_peak_price': float(p2_price),
                    'neckline_price': float(neckline_price),
                    'target_price': float(target_price),
                    'pattern_height': float(pattern_height),
                    'confidence': confidence,
                    'status': 'confirmed' if breakdown_idx else 'forming',
                    'duration_days': duration,
                    'pullback_strength': pullback,
                    'm_shape_score': m_shape_score,
                    'volume_confirmation': volume_confirmation,
                    'detection_method': 'traditional_enhanced'
                }
                
                patterns.append(pattern)
        
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def detect_cup_and_handle(self, df: pd.DataFrame) -> List[Dict]:
        """Enhanced Cup and Handle detection"""
        if not validate_dataframe(df) or len(df) < 100:
            return []
        
        prices = df['Close'].values
        dates = df['Date'] if 'Date' in df.columns else df.index
        peaks, troughs = self.find_peaks_troughs(prices, order=8)
        
        if len(peaks) < 2 or len(troughs) < 2:
            return []
        
        patterns = []
        
        for i in range(len(peaks) - 1):
            left_rim_idx = peaks[i]
            left_rim_price = prices[left_rim_idx]
            
            # Find cup bottom
            cup_candidates = [t for t in troughs if left_rim_idx < t < left_rim_idx + 120]
            if not cup_candidates:
                continue
                
            cup_bottom_idx = min(cup_candidates, key=lambda x: prices[x])
            cup_bottom_price = prices[cup_bottom_idx]
            
            # Cup depth check
            cup_depth = (left_rim_price - cup_bottom_price) / left_rim_price
            if cup_depth < self.config.min_cup_depth:
                continue
            
            # Find right rim
            right_rim_candidates = [p for p in peaks if cup_bottom_idx < p < cup_bottom_idx + 80]
            if not right_rim_candidates:
                continue
            
            valid_right_rims = []
            for candidate in right_rim_candidates:
                candidate_price = prices[candidate]
                rim_diff = abs(candidate_price - left_rim_price) / left_rim_price
                if rim_diff <= self.config.rim_tolerance:
                    valid_right_rims.append(candidate)
            
            if not valid_right_rims:
                continue
            
            right_rim_idx = valid_right_rims[0]
            right_rim_price = prices[right_rim_idx]
            
            # Find handle
            handle_candidates = [t for t in troughs if right_rim_idx < t < right_rim_idx + 40]
            if not handle_candidates:
                continue
            
            handle_bottom_idx = handle_candidates[0]
            handle_bottom_price = prices[handle_bottom_idx]
            
            # Handle depth check
            cup_height = left_rim_price - cup_bottom_price
            handle_retrace = (right_rim_price - handle_bottom_price) / cup_height
            
            if not (self.config.handle_retrace_min <= handle_retrace <= self.config.handle_retrace_max):
                continue
            
            duration = (dates.iloc[handle_bottom_idx] - dates.iloc[left_rim_idx]).days
            if duration < 60:  # Minimum duration for cup and handle
                continue
            
            confidence = self._calculate_ch_confidence(cup_depth, handle_retrace, 
                                                     abs(left_rim_price - right_rim_price) / left_rim_price)
            
            if confidence < 0.5:
                continue
            
            rim_level = max(left_rim_price, right_rim_price)
            breakout_idx = self._check_breakout(df, handle_bottom_idx, rim_level, 'up')
            
            pattern = {
                'type': 'Cup and Handle',
                'left_rim_idx': int(left_rim_idx),
                'cup_bottom_idx': int(cup_bottom_idx),
                'right_rim_idx': int(right_rim_idx),
                'handle_bottom_idx': int(handle_bottom_idx),
                'breakout_idx': breakout_idx,
                'left_rim_price': float(left_rim_price),
                'cup_bottom_price': float(cup_bottom_price),
                'right_rim_price': float(right_rim_price),
                'handle_bottom_price': float(handle_bottom_price),
                'rim_level': float(rim_level),
                'target_price': float(rim_level + cup_height),
                'cup_height': float(cup_height),
                'cup_depth': cup_depth,
                'handle_retrace': handle_retrace,
                'confidence': confidence,
                'status': 'confirmed' if breakout_idx else 'forming',
                'duration_days': duration,
                'detection_method': 'traditional'
            }
            
            patterns.append(pattern)
        
        return sorted(patterns, key=lambda x: x['confidence'], reverse=True)[:3]
    
    def _calculate_hs_confidence(self, shoulder_diff: float, neckline_diff: float, 
                                head_price: float, neckline: float, duration: int) -> float:
        """Calculate Head and Shoulders pattern confidence"""
        confidence = 0.6
        
        # Shoulder symmetry
        confidence += (1 - min(1.0, shoulder_diff / self.config.shoulder_tolerance)) * 0.15
        
        # Neckline quality
        confidence += (1 - min(1.0, neckline_diff / self.config.neckline_tolerance)) * 0.10
        
        # Pattern height (stronger patterns are more significant)
        height_ratio = (head_price - neckline) / neckline
        confidence += min(0.10, height_ratio * 0.5)
        
        # Duration bonus
        if duration > 60:
            confidence += 0.05
        
        return min(0.95, confidence)
    
    def _calculate_db_confidence(self, price_diff: float, rebound: float, duration: int) -> float:
        """Calculate Double Bottom confidence"""
        confidence = 0.6
        confidence += (1 - min(1.0, price_diff / self.config.price_tolerance)) * 0.15
        confidence += min(0.15, rebound * 2)
        if duration > 60:
            confidence += 0.05
        return min(0.95, confidence)
    
    def _calculate_dt_confidence(self, price_diff: float, pullback: float, duration: int) -> float:
        """Calculate Double Top confidence"""
        confidence = 0.6
        confidence += (1 - min(1.0, price_diff / self.config.price_tolerance)) * 0.15
        confidence += min(0.15, pullback * 2)
        if duration > 60:
            confidence += 0.05
        return min(0.95, confidence)
    
    def _calculate_ch_confidence(self, cup_depth: float, handle_retrace: float, rim_diff: float) -> float:
        """Calculate Cup and Handle confidence"""
        confidence = 0.5
        confidence += min(0.20, cup_depth * 0.8)
        confidence += (1 - abs(handle_retrace - 0.3) / 0.2) * 0.15
        confidence += (1 - min(1.0, rim_diff / self.config.rim_tolerance)) * 0.10
        return min(0.95, confidence)
    
    def _check_breakout(self, df: pd.DataFrame, start_idx: int, level: float, 
                       direction: str, lookforward: int = 20) -> Optional[int]:
        """Check for breakout/breakdown from pattern"""
        if start_idx + lookforward >= len(df):
            lookforward = len(df) - start_idx - 1
        
        for i in range(start_idx + 1, start_idx + lookforward + 1):
            price = df['Close'].iloc[i]
            
            if direction == 'up' and price > level * (1 + self.config.breakout_threshold):
                return int(i)
            elif direction == 'down' and price < level * (1 - self.config.breakout_threshold):
                return int(i)
        
        return None
    
    def _check_enhanced_breakout(self, df: pd.DataFrame, start_idx: int, level: float, 
                                direction: str, lookforward: int = 20) -> Optional[int]:
        """Enhanced breakout detection with volume confirmation"""
        if start_idx + lookforward >= len(df):
            lookforward = len(df) - start_idx - 1
        
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        
        for i in range(start_idx + 1, start_idx + lookforward + 1):
            price = df['Close'].iloc[i]
            volume = volumes[i] if volumes is not None else 0
            
            # Check for breakout with volume confirmation
            if direction == 'up' and price > level * (1 + self.config.breakout_threshold):
                # Volume should be above average for confirmation
                if volumes is not None:
                    avg_volume = np.mean(volumes[max(0, i-10):i])
                    if volume > avg_volume * 0.8:  # Allow some flexibility
                        return int(i)
                else:
                    return int(i)
            elif direction == 'down' and price < level * (1 - self.config.breakout_threshold):
                # Volume should be above average for confirmation
                if volumes is not None:
                    avg_volume = np.mean(volumes[max(0, i-10):i])
                    if volume > avg_volume * 0.8:  # Allow some flexibility
                        return int(i)
                else:
                    return int(i)
        
        return None
    
    def _analyze_w_shape(self, prices: np.ndarray, t1_idx: int, t2_idx: int, neckline_idx: int) -> float:
        """Analyze W-shape formation for double bottom patterns"""
        try:
            # Extract the pattern segment
            start_idx = max(0, t1_idx - 5)
            end_idx = min(len(prices), t2_idx + 5)
            pattern_segment = prices[start_idx:end_idx]
            
            # Calculate symmetry around the neckline
            left_side = pattern_segment[:neckline_idx-start_idx]
            right_side = pattern_segment[neckline_idx-start_idx:]
            
            if len(left_side) < 3 or len(right_side) < 3:
                return 0.5
            
            # Check for W-shape characteristics
            # 1. Both troughs should be at similar levels
            t1_price = prices[t1_idx]
            t2_price = prices[t2_idx]
            trough_similarity = 1 - abs(t1_price - t2_price) / max(t1_price, t2_price)
            
            # 2. Neckline should be higher than both troughs
            neckline_price = prices[neckline_idx]
            neckline_quality = min(1.0, (neckline_price - min(t1_price, t2_price)) / max(t1_price, t2_price) * 2)
            
            # 3. Pattern should be roughly symmetrical
            left_length = len(left_side)
            right_length = len(right_side)
            symmetry = 1 - abs(left_length - right_length) / max(left_length, right_length)
            
            # Combine scores
            w_score = (trough_similarity * 0.4 + neckline_quality * 0.4 + symmetry * 0.2)
            return max(0.1, min(1.0, w_score))
            
        except Exception:
            return 0.5
    
    def _analyze_m_shape(self, prices: np.ndarray, p1_idx: int, p2_idx: int, neckline_idx: int) -> float:
        """Analyze M-shape formation for double top patterns"""
        try:
            # Extract the pattern segment
            start_idx = max(0, p1_idx - 5)
            end_idx = min(len(prices), p2_idx + 5)
            pattern_segment = prices[start_idx:end_idx]
            
            # Calculate symmetry around the neckline
            left_side = pattern_segment[:neckline_idx-start_idx]
            right_side = pattern_segment[neckline_idx-start_idx:]
            
            if len(left_side) < 3 or len(right_side) < 3:
                return 0.5
            
            # Check for M-shape characteristics
            # 1. Both peaks should be at similar levels
            p1_price = prices[p1_idx]
            p2_price = prices[p2_idx]
            peak_similarity = 1 - abs(p1_price - p2_price) / max(p1_price, p2_price)
            
            # 2. Neckline should be lower than both peaks
            neckline_price = prices[neckline_idx]
            neckline_quality = min(1.0, (max(p1_price, p2_price) - neckline_price) / max(p1_price, p2_price) * 2)
            
            # 3. Pattern should be roughly symmetrical
            left_length = len(left_side)
            right_length = len(right_side)
            symmetry = 1 - abs(left_length - right_length) / max(left_length, right_length)
            
            # Combine scores
            m_score = (peak_similarity * 0.4 + neckline_quality * 0.4 + symmetry * 0.2)
            return max(0.1, min(1.0, m_score))
            
        except Exception:
            return 0.5
    
    def _analyze_volume_pattern(self, volumes: np.ndarray, t1_idx: int, t2_idx: int, neckline_idx: int) -> float:
        """Analyze volume pattern for double bottom (bullish)"""
        try:
            # Ideal volume pattern: higher volume at second trough, lower at neckline
            t1_volume = volumes[t1_idx] if t1_idx < len(volumes) else 0
            t2_volume = volumes[t2_idx] if t2_idx < len(volumes) else 0
            neckline_volume = volumes[neckline_idx] if neckline_idx < len(volumes) else 0
            
            # Calculate average volume for comparison
            start_idx = max(0, t1_idx - 10)
            end_idx = min(len(volumes), t2_idx + 10)
            avg_volume = np.mean(volumes[start_idx:end_idx])
            
            # Score based on volume characteristics
            score = 0.5  # Base score
            
            # Higher volume at second trough is bullish
            if t2_volume > t1_volume:
                score += 0.2
            
            # Volume should be above average at troughs
            if t1_volume > avg_volume * 0.8 and t2_volume > avg_volume * 0.8:
                score += 0.2
            
            # Lower volume at neckline (consolidation)
            if neckline_volume < avg_volume * 1.2:
                score += 0.1
            
            return max(0.1, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _analyze_volume_pattern_bearish(self, volumes: np.ndarray, p1_idx: int, p2_idx: int, neckline_idx: int) -> float:
        """Analyze volume pattern for double top (bearish)"""
        try:
            # Ideal volume pattern: lower volume at second peak, higher at neckline
            p1_volume = volumes[p1_idx] if p1_idx < len(volumes) else 0
            p2_volume = volumes[p2_idx] if p2_idx < len(volumes) else 0
            neckline_volume = volumes[neckline_idx] if neckline_idx < len(volumes) else 0
            
            # Calculate average volume for comparison
            start_idx = max(0, p1_idx - 10)
            end_idx = min(len(volumes), p2_idx + 10)
            avg_volume = np.mean(volumes[start_idx:end_idx])
            
            # Score based on volume characteristics
            score = 0.5  # Base score
            
            # Lower volume at second peak is bearish
            if p2_volume < p1_volume:
                score += 0.2
            
            # Volume should be above average at peaks
            if p1_volume > avg_volume * 0.8 and p2_volume > avg_volume * 0.8:
                score += 0.2
            
            # Higher volume at neckline (distribution)
            if neckline_volume > avg_volume * 0.8:
                score += 0.1
            
            return max(0.1, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _validate_support_level(self, prices: np.ndarray, t1_idx: int, t2_idx: int) -> float:
        """Validate support level strength for double bottom"""
        try:
            t1_price = prices[t1_idx]
            t2_price = prices[t2_idx]
            support_level = min(t1_price, t2_price)
            
            # Check how many times price touched this level
            tolerance = support_level * 0.02  # 2% tolerance
            touches = 0
            
            for i in range(max(0, t1_idx - 20), min(len(prices), t2_idx + 20)):
                if abs(prices[i] - support_level) <= tolerance:
                    touches += 1
            
            # More touches indicate stronger support
            if touches >= 3:
                return 1.0
            elif touches >= 2:
                return 0.8
            else:
                return 0.6
                
        except Exception:
            return 0.5
    
    def _validate_resistance_level(self, prices: np.ndarray, p1_idx: int, p2_idx: int) -> float:
        """Validate resistance level strength for double top"""
        try:
            p1_price = prices[p1_idx]
            p2_price = prices[p2_idx]
            resistance_level = max(p1_price, p2_price)
            
            # Check how many times price touched this level
            tolerance = resistance_level * 0.02  # 2% tolerance
            touches = 0
            
            for i in range(max(0, p1_idx - 20), min(len(prices), p2_idx + 20)):
                if abs(prices[i] - resistance_level) <= tolerance:
                    touches += 1
            
            # More touches indicate stronger resistance
            if touches >= 3:
                return 1.0
            elif touches >= 2:
                return 0.8
            else:
                return 0.6
                
        except Exception:
            return 0.5
    
    def _calculate_enhanced_db_confidence(self, price_diff: float, rebound: float, duration: int,
                                        w_shape_score: float, volume_pattern_score: float,
                                        support_level_score: float, volume_confirmation: bool) -> float:
        """Calculate enhanced Double Bottom confidence"""
        confidence = 0.5  # Base confidence
        
        # Price similarity
        confidence += (1 - min(1.0, price_diff / self.config.price_tolerance)) * 0.15
        
        # Rebound strength
        confidence += min(0.15, rebound * 2)
        
        # Duration bonus
        if duration > 60:
            confidence += 0.05
        
        # W-shape formation
        confidence += w_shape_score * 0.15
        
        # Volume pattern
        confidence += volume_pattern_score * 0.10
        
        # Support level validation
        confidence += support_level_score * 0.10
        
        # Volume confirmation bonus
        if volume_confirmation:
            confidence += 0.05
        
        return max(0.1, min(0.95, confidence))
    
    def _calculate_enhanced_dt_confidence(self, price_diff: float, pullback: float, duration: int,
                                        m_shape_score: float, volume_pattern_score: float,
                                        resistance_level_score: float, volume_confirmation: bool) -> float:
        """Calculate enhanced Double Top confidence"""
        confidence = 0.5  # Base confidence
        
        # Price similarity
        confidence += (1 - min(1.0, price_diff / self.config.price_tolerance)) * 0.15
        
        # Pullback strength
        confidence += min(0.15, pullback * 2)
        
        # Duration bonus
        if duration > 60:
            confidence += 0.05
        
        # M-shape formation
        confidence += m_shape_score * 0.15
        
        # Volume pattern
        confidence += volume_pattern_score * 0.10
        
        # Resistance level validation
        confidence += resistance_level_score * 0.10
        
        # Volume confirmation bonus
        if volume_confirmation:
            confidence += 0.05
        
        return max(0.1, min(0.95, confidence))
    
    def _analyze_hs_volume_pattern(self, volumes: np.ndarray, ls_idx: int, head_idx: int, rs_idx: int,
                                 left_trough_idx: int, right_trough_idx: int) -> Tuple[bool, float]:
        """Analyze volume pattern for Head and Shoulders (bearish)"""
        try:
            # Ideal volume pattern: higher volume at head, lower at right shoulder
            ls_volume = volumes[ls_idx] if ls_idx < len(volumes) else 0
            head_volume = volumes[head_idx] if head_idx < len(volumes) else 0
            rs_volume = volumes[rs_idx] if rs_idx < len(volumes) else 0
            
            # Calculate average volume for comparison
            start_idx = max(0, ls_idx - 10)
            end_idx = min(len(volumes), rs_idx + 10)
            avg_volume = np.mean(volumes[start_idx:end_idx])
            
            # Score based on volume characteristics
            score = 0.5  # Base score
            
            # Higher volume at head is bearish (distribution)
            if head_volume > ls_volume and head_volume > rs_volume:
                score += 0.2
            
            # Lower volume at right shoulder (exhaustion)
            if rs_volume < head_volume:
                score += 0.2
            
            # Volume should be above average at peaks
            if ls_volume > avg_volume * 0.8 and head_volume > avg_volume * 0.8 and rs_volume > avg_volume * 0.8:
                score += 0.1
            
            # Volume confirmation
            volume_confirmation = (head_volume > ls_volume * 0.8 and rs_volume < head_volume * 1.2)
            
            return volume_confirmation, max(0.1, min(1.0, score))
            
        except Exception:
            return True, 0.5
    
    def _analyze_hs_symmetry(self, prices: np.ndarray, ls_idx: int, head_idx: int, rs_idx: int) -> float:
        """Analyze symmetry of Head and Shoulders pattern"""
        try:
            # Calculate distances and price levels
            left_distance = head_idx - ls_idx
            right_distance = rs_idx - head_idx
            
            # Symmetry score based on distance ratio
            if left_distance > 0 and right_distance > 0:
                distance_ratio = min(left_distance, right_distance) / max(left_distance, right_distance)
            else:
                distance_ratio = 0
            
            # Price symmetry (shoulders should be at similar levels)
            ls_price = prices[ls_idx]
            rs_price = prices[rs_idx]
            price_symmetry = 1 - abs(ls_price - rs_price) / max(ls_price, rs_price)
            
            # Combined symmetry score
            symmetry_score = (distance_ratio * 0.6 + price_symmetry * 0.4)
            return max(0.1, min(1.0, symmetry_score))
            
        except Exception:
            return 0.5
    
    def _validate_hs_support_level(self, prices: np.ndarray, left_trough_idx: int, right_trough_idx: int) -> float:
        """Validate support level strength for Head and Shoulders"""
        try:
            left_trough_price = prices[left_trough_idx]
            right_trough_price = prices[right_trough_idx]
            support_level = (left_trough_price + right_trough_price) / 2
            
            # Check how many times price touched this level
            tolerance = support_level * 0.02  # 2% tolerance
            touches = 0
            
            for i in range(max(0, left_trough_idx - 20), min(len(prices), right_trough_idx + 20)):
                if abs(prices[i] - support_level) <= tolerance:
                    touches += 1
            
            # More touches indicate stronger support
            if touches >= 3:
                return 1.0
            elif touches >= 2:
                return 0.8
            else:
                return 0.6
                
        except Exception:
            return 0.5
    
    def _calculate_enhanced_hs_confidence(self, shoulder_diff: float, neckline_diff: float,
                                        head_price: float, neckline_level: float, duration: int,
                                        volume_pattern_score: float, symmetry_score: float,
                                        support_level_score: float, volume_confirmation: bool) -> float:
        """Calculate enhanced Head and Shoulders confidence"""
        confidence = 0.5  # Base confidence
        
        # Shoulder symmetry
        confidence += (1 - min(1.0, shoulder_diff / self.config.shoulder_tolerance)) * 0.15
        
        # Neckline quality
        confidence += (1 - min(1.0, neckline_diff / self.config.neckline_tolerance)) * 0.10
        
        # Pattern height (stronger patterns are more significant)
        height_ratio = (head_price - neckline_level) / neckline_level
        confidence += min(0.10, height_ratio * 0.5)
        
        # Duration bonus
        if duration > 60:
            confidence += 0.05
        
        # Symmetry analysis
        confidence += symmetry_score * 0.10
        
        # Volume pattern
        confidence += volume_pattern_score * 0.10
        
        # Support level validation
        confidence += support_level_score * 0.10
        
        # Volume confirmation bonus
        if volume_confirmation:
            confidence += 0.05
        
        return max(0.1, min(0.95, confidence))

# =========================== IMPROVED ML DETECTOR ===========================

class ImprovedMLDetector:
    """Enhanced ML pattern detector with better feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = None
        self.svm_model = None
        self.is_trained = False
        self.feature_names = []
        
    def extract_technical_features(self, df: pd.DataFrame, window_size: int = 20) -> np.ndarray:
        """Extract comprehensive technical features"""
        try:
            if len(df) < window_size + 20:
                return np.array([])
            
            data = df.copy()
            features = []
            
            # Ensure required columns exist
            required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    return np.array([])
            
            # Price-based features
            data['returns'] = data['Close'].pct_change()
            data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['volatility'] = data['returns'].rolling(window_size).std()
            
            # Safe division for price_position
            high_max = data['High'].rolling(window_size).max()
            low_min = data['Low'].rolling(window_size).min()
            price_range = high_max - low_min
            data['price_position'] = np.where(
                price_range > 0,
                (data['Close'] - low_min) / price_range,
                0.5  # Default to middle if no range
            )
            
            # Moving averages and ratios
            for period in [5, 10, 20]:
                data[f'ma_{period}'] = data['Close'].rolling(period).mean()
                # Safe division for ma_ratio
                data[f'ma_ratio_{period}'] = np.where(
                    data[f'ma_{period}'] > 0,
                    data['Close'] / data[f'ma_{period}'],
                    1.0  # Default to 1.0 if moving average is 0
                )
                data[f'ma_slope_{period}'] = data[f'ma_{period}'].diff(5)
            
            # Technical indicators
            # RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            # Safe division for RSI
            data['rsi'] = np.where(
                loss > 0,
                100 - (100 / (1 + gain / loss)),
                50  # Default to 50 if no loss
            )
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # Bollinger Bands
            data['bb_middle'] = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            # Safe division for bb_width and bb_position
            bb_range = data['bb_upper'] - data['bb_lower']
            data['bb_width'] = np.where(
                data['bb_middle'] > 0,
                bb_range / data['bb_middle'],
                0
            )
            data['bb_position'] = np.where(
                bb_range > 0,
                (data['Close'] - data['bb_lower']) / bb_range,
                0.5
            )
            
            # Volume features
            data['volume_sma'] = data['Volume'].rolling(20).mean()
            data['volume_ratio'] = np.where(
                data['volume_sma'] > 0,
                data['Volume'] / data['volume_sma'],
                1.0
            )
            
            # Pattern-specific features
            peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
            troughs = argrelextrema(data['Close'].values, np.less, order=5)[0]
            
            data['is_peak'] = 0
            data['is_trough'] = 0
            if len(peaks) > 0:
                data.iloc[peaks, data.columns.get_loc('is_peak')] = 1
            if len(troughs) > 0:
                data.iloc[troughs, data.columns.get_loc('is_trough')] = 1
            
            # Rolling statistics
            data['high_low_ratio'] = np.where(
                data['Low'] > 0,
                data['High'] / data['Low'],
                1.0
            )
            data['close_open_ratio'] = np.where(
                data['Open'] > 0,
                data['Close'] / data['Open'],
                1.0
            )
            
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Extract windowed features
            feature_cols = [
                'returns', 'log_returns', 'volatility', 'price_position',
                'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20',
                'ma_slope_5', 'ma_slope_10', 'ma_slope_20',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_width', 'bb_position', 'volume_ratio',
                'is_peak', 'is_trough', 'high_low_ratio', 'close_open_ratio'
            ]
            
            # Create windowed features
            for i in range(window_size, len(data)):
                window_features = []
                for col in feature_cols:
                    if col in data.columns:
                        window_data = data[col].iloc[i-window_size:i]
                        # Statistical moments
                        window_features.extend([
                            window_data.mean(),
                            window_data.std(),
                            window_data.min(),
                            window_data.max(),
                            window_data.skew() if len(window_data) > 2 else 0,
                            window_data.iloc[-1] - window_data.iloc[0]  # Trend
                        ])
                
                if len(window_features) == len(feature_cols) * 6:
                    features.append(window_features)
            
            return np.array(features)
            
        except Exception as e:
            logging.error(f"Error in extract_technical_features: {str(e)}")
            return np.array([])
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data with more realistic patterns"""
        X, y = [], []
        
        for pattern_type in range(4):
            for _ in range(n_samples // 4):
                if pattern_type == 0:  # Head and Shoulders
                    pattern_data = self._generate_head_shoulders()
                elif pattern_type == 1:  # Double Bottom
                    pattern_data = self._generate_double_bottom()
                elif pattern_type == 2:  # Double Top
                    pattern_data = self._generate_double_top()
                else:  # Cup and Handle
                    pattern_data = self._generate_cup_handle()
                
                # Extract features from synthetic data
                df_synthetic = pd.DataFrame({
                    'Close': pattern_data,
                    'Open': pattern_data * (0.99 + np.random.random(len(pattern_data)) * 0.02),
                    'High': pattern_data * (1.01 + np.random.random(len(pattern_data)) * 0.02),
                    'Low': pattern_data * (0.98 + np.random.random(len(pattern_data)) * 0.02),
                    'Volume': np.random.lognormal(10, 0.5, len(pattern_data))
                })
                
                features = self._extract_pattern_features(df_synthetic)
                if len(features) > 0:
                    X.append(features)
                    y.append(pattern_type)
        
        return np.array(X), np.array(y)
    
    def _extract_pattern_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract statistical features from pattern data (22 features for ML compatibility)"""
        if len(df) < 30:
            return np.array([])
        
        prices = df['Close'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else None
        
        # Basic statistical features (7 features)
        mean_price = np.mean(prices)
        features = [
            np.mean(prices), np.std(prices), np.min(prices), np.max(prices),
            np.percentile(prices, 25), np.percentile(prices, 75),
            (np.max(prices) - np.min(prices)) / mean_price if mean_price > 0 else 0,  # Price range ratio
        ]
        
        # Price movement features (5 features)
        # Safe division for returns
        price_changes = np.diff(prices)
        returns = np.where(prices[:-1] > 0, price_changes / prices[:-1], 0)
        features.extend([
            np.mean(returns), np.std(returns), 
            np.sum(returns > 0) / len(returns),  # Positive return ratio
            np.max(returns), np.min(returns)
        ])
        
        # Trend features (1 feature)
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        mean_price = np.mean(prices)
        features.append(slope / mean_price if mean_price > 0 else 0)  # Normalized slope
        
        # Peak and trough features (4 features)
        peaks = argrelextrema(prices, np.greater, order=3)[0]
        troughs = argrelextrema(prices, np.less, order=3)[0]
        
        features.extend([
            len(peaks) / len(prices),  # Peak density
            len(troughs) / len(prices),  # Trough density
            np.std(prices[peaks]) if len(peaks) > 1 else 0,  # Peak volatility
            np.std(prices[troughs]) if len(troughs) > 1 else 0,  # Trough volatility
        ])
        
        # Symmetry features (2 features)
        mid_point = len(prices) // 2
        left_half = prices[:mid_point]
        right_half = prices[mid_point:]
        
        if len(left_half) > 0 and len(right_half) > 0:
            features.extend([
                np.corrcoef(left_half, right_half[:len(left_half)])[0, 1] if len(right_half) >= len(left_half) else 0,
                abs(np.mean(left_half) - np.mean(right_half)) / mean_price if mean_price > 0 else 0
            ])
        else:
            features.extend([0, 0])
        
        # Volume features (3 features)
        if volumes is not None:
            features.extend([
                np.mean(volumes), np.std(volumes),
                np.corrcoef(prices, volumes)[0, 1] if len(prices) == len(volumes) else 0
            ])
        else:
            features.extend([0, 0, 0])
        
        # Total: 7 + 5 + 1 + 4 + 2 + 3 = 22 features
        return np.array(features)
    
    def _calculate_w_m_score(self, prices: np.ndarray, pattern_type: str) -> float:
        """Calculate W or M shape score for pattern detection"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Normalize prices to 0-1 range
            min_price = np.min(prices)
            max_price = np.max(prices)
            if max_price == min_price:
                return 0.5
            
            normalized_prices = (prices - min_price) / (max_price - min_price)
            
            # Find peaks and troughs
            peaks = argrelextrema(normalized_prices, np.greater, order=2)[0]
            troughs = argrelextrema(normalized_prices, np.less, order=2)[0]
            
            if pattern_type == 'w':
                # W pattern: two troughs with a peak in between
                if len(troughs) >= 2 and len(peaks) >= 1:
                    # Check if there's a peak between the first two troughs
                    if len(troughs) >= 2:
                        t1, t2 = troughs[0], troughs[1]
                        between_peaks = [p for p in peaks if t1 < p < t2]
                        if between_peaks:
                            # Calculate W score based on trough similarity and peak height
                            t1_val = normalized_prices[t1]
                            t2_val = normalized_prices[t2]
                            peak_val = normalized_prices[between_peaks[0]]
                            
                            trough_similarity = 1 - abs(t1_val - t2_val)
                            peak_height = peak_val - min(t1_val, t2_val)
                            
                            return (trough_similarity * 0.6 + peak_height * 0.4)
            
            elif pattern_type == 'm':
                # M pattern: two peaks with a trough in between
                if len(peaks) >= 2 and len(troughs) >= 1:
                    # Check if there's a trough between the first two peaks
                    if len(peaks) >= 2:
                        p1, p2 = peaks[0], peaks[1]
                        between_troughs = [t for t in troughs if p1 < t < p2]
                        if between_troughs:
                            # Calculate M score based on peak similarity and trough depth
                            p1_val = normalized_prices[p1]
                            p2_val = normalized_prices[p2]
                            trough_val = normalized_prices[between_troughs[0]]
                            
                            peak_similarity = 1 - abs(p1_val - p2_val)
                            trough_depth = max(p1_val, p2_val) - trough_val
                            
                            return (peak_similarity * 0.6 + trough_depth * 0.4)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _calculate_support_resistance_score(self, prices: np.ndarray, level_type: str) -> float:
        """Calculate support or resistance level strength"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Find potential support/resistance levels
            if level_type == 'support':
                # Look for local minima
                troughs = argrelextrema(prices, np.less, order=3)[0]
                if len(troughs) < 2:
                    return 0.5
                
                # Check if troughs are at similar levels
                trough_prices = prices[troughs]
                min_trough = np.min(trough_prices)
                max_trough = np.max(trough_prices)
                
                if max_trough == min_trough:
                    return 1.0
                
                # Calculate how close the troughs are
                similarity = 1 - (max_trough - min_trough) / max_trough
                return max(0.1, min(1.0, similarity))
            
            elif level_type == 'resistance':
                # Look for local maxima
                peaks = argrelextrema(prices, np.greater, order=3)[0]
                if len(peaks) < 2:
                    return 0.5
                
                # Check if peaks are at similar levels
                peak_prices = prices[peaks]
                min_peak = np.min(peak_prices)
                max_peak = np.max(peak_prices)
                
                if max_peak == min_peak:
                    return 1.0
                
                # Calculate how close the peaks are
                similarity = 1 - (max_peak - min_peak) / max_peak
                return max(0.1, min(1.0, similarity))
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _generate_head_shoulders(self, length: int = 50) -> np.ndarray:
        """Generate realistic head and shoulders pattern with volume characteristics"""
        pattern = np.ones(length) * 100
        
        # Left shoulder (days 8-18)
        ls_start, ls_end = 8, 18
        pattern[ls_start:ls_end] = 100 + 8 * np.sin(np.linspace(0, np.pi, ls_end - ls_start))
        
        # Decline to left trough
        decline1_start, decline1_end = 18, 22
        pattern[decline1_start:decline1_end] = np.linspace(108, 95, decline1_end - decline1_start)
        
        # Head formation (days 22-32)
        head_start, head_end = 22, 32
        pattern[head_start:head_end] = 95 + 15 * np.sin(np.linspace(0, np.pi, head_end - head_start))
        
        # Decline to right trough
        decline2_start, decline2_end = 32, 36
        pattern[decline2_start:decline2_end] = np.linspace(110, 95, decline2_end - decline2_start)
        
        # Right shoulder (days 36-46)
        if length > 40:
            rs_start, rs_end = 36, min(46, length)
            pattern[rs_start:rs_end] = 95 + 8 * np.sin(np.linspace(0, np.pi, rs_end - rs_start))
        
        # Breakdown with momentum
        if length > 46:
            breakdown_start = 46
            # Strong downward move with measured move target
            pattern[breakdown_start:] = np.linspace(103, 85, length - breakdown_start)
        
        # Add realistic noise (higher at peaks, lower at troughs)
        noise = np.random.normal(0, 1.0, length)
        # Increase noise at peaks to simulate higher volume
        noise[ls_start:ls_end] *= 1.3
        noise[head_start:head_end] *= 1.5  # Highest volume at head
        noise[rs_start:rs_end] *= 1.2
        
        return pattern + noise
    
    def _generate_double_bottom(self, length: int = 50) -> np.ndarray:
        """Generate realistic double bottom pattern with volume characteristics"""
        pattern = np.ones(length) * 100
        
        # First bottom (support level)
        bottom1_start, bottom1_end = 12, 18
        pattern[bottom1_start:bottom1_end] = 90 + 5 * np.sin(np.linspace(np.pi, 2*np.pi, bottom1_end - bottom1_start))
        
        # Rebound to neckline
        rebound_start, rebound_end = 18, 25
        pattern[rebound_start:rebound_end] = np.linspace(90, 105, rebound_end - rebound_start)
        
        # Peak between bottoms (neckline)
        peak_start, peak_end = 25, 30
        pattern[peak_start:peak_end] = 105 + 2 * np.sin(np.linspace(0, np.pi, peak_end - peak_start))
        
        # Decline to second bottom
        decline_start, decline_end = 30, 37
        pattern[decline_start:decline_end] = np.linspace(105, 90, decline_end - decline_start)
        
        # Second bottom (stronger support)
        bottom2_start, bottom2_end = 37, 42
        pattern[bottom2_start:bottom2_end] = 90 + 3 * np.sin(np.linspace(np.pi, 2*np.pi, bottom2_end - bottom2_start))
        
        # Breakout with momentum
        if length > 42:
            breakout_start = 42
            # Strong upward move with measured move target
            pattern[breakout_start:] = np.linspace(90, 120, length - breakout_start)
        
        # Add realistic noise (higher at bottoms, lower at peaks)
        noise = np.random.normal(0, 1.0, length)
        # Increase noise at bottoms to simulate higher volume
        noise[bottom1_start:bottom1_end] *= 1.5
        noise[bottom2_start:bottom2_end] *= 1.5
        
        return pattern + noise
    
    def _generate_double_top(self, length: int = 50) -> np.ndarray:
        """Generate realistic double top pattern with volume characteristics"""
        pattern = np.ones(length) * 100
        
        # First peak (resistance level)
        peak1_start, peak1_end = 12, 18
        pattern[peak1_start:peak1_end] = 115 + 5 * np.sin(np.linspace(0, np.pi, peak1_end - peak1_start))
        
        # Decline to neckline
        decline_start, decline_end = 18, 25
        pattern[decline_start:decline_end] = np.linspace(115, 95, decline_end - decline_start)
        
        # Trough between peaks (neckline)
        trough_start, trough_end = 25, 30
        pattern[trough_start:trough_end] = 95 + 2 * np.sin(np.linspace(np.pi, 2*np.pi, trough_end - trough_start))
        
        # Rally to second peak
        rally_start, rally_end = 30, 37
        pattern[rally_start:rally_end] = np.linspace(95, 115, rally_end - rally_start)
        
        # Second peak (weaker resistance)
        peak2_start, peak2_end = 37, 42
        pattern[peak2_start:peak2_end] = 115 + 3 * np.sin(np.linspace(0, np.pi, peak2_end - peak2_start))
        
        # Breakdown with momentum
        if length > 42:
            breakdown_start = 42
            # Strong downward move with measured move target
            pattern[breakdown_start:] = np.linspace(115, 85, length - breakdown_start)
        
        # Add realistic noise (higher at peaks, lower at troughs)
        noise = np.random.normal(0, 1.0, length)
        # Increase noise at peaks to simulate higher volume
        noise[peak1_start:peak1_end] *= 1.5
        noise[peak2_start:peak2_end] *= 1.5
        
        return pattern + noise
    
    def _generate_cup_handle(self, length: int = 60) -> np.ndarray:
        """Generate realistic cup and handle pattern"""
        pattern = np.ones(length) * 100
        
        # Cup formation (parabolic)
        cup_start, cup_end = 10, 45
        cup_length = cup_end - cup_start
        x = np.linspace(-2, 2, cup_length)
        cup_shape = 100 - 15 * (x ** 2) + 15  # Parabolic cup
        pattern[cup_start:cup_end] = cup_shape
        
        # Handle formation
        if length > 50:
            handle_start, handle_end = 48, min(58, length)
            handle_length = handle_end - handle_start
            handle_shape = np.linspace(100, 95, handle_length // 2)
            handle_shape = np.concatenate([handle_shape, np.linspace(95, 100, handle_length - len(handle_shape))])
            pattern[handle_start:handle_end] = handle_shape
            
            # Breakout
            if length > handle_end:
                pattern[handle_end:] = np.linspace(100, 110, length - handle_end)
        
        # Add noise
        noise = np.random.normal(0, 2, length)
        return pattern + noise
    
    def train_models(self) -> bool:
        """Train ML models with improved data"""
        try:
            st.info("🤖 Training ML models...")
            progress = st.progress(0)
            
            # Generate training data
            progress.progress(0.2)
            X, y = self.generate_synthetic_data(2000)
            
            if len(X) == 0 or len(y) == 0:
                st.error("Failed to generate training data")
                progress.empty()
                return False
            
            # Ensure we have enough data
            if len(X) < 100:
                st.error("Insufficient training data generated")
                progress.empty()
                return False
            
            progress.progress(0.4)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                st.error("Training data contains NaN or infinite values")
                progress.empty()
                return False
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            progress.progress(0.6)
            
            # Train Random Forest
            self.rf_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            
            progress.progress(0.8)
            
            # Train SVM
            self.svm_model = SVC(
                probability=True,
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            )
            self.svm_model.fit(X_train_scaled, y_train)
            
            progress.progress(1.0)
            
            # Evaluate models
            rf_score = self.rf_model.score(X_test, y_test)
            svm_score = self.svm_model.score(X_test_scaled, y_test)
            
            st.success(f"✅ ML models trained! RF: {rf_score:.2%}, SVM: {svm_score:.2%}")
            
            self.is_trained = True
            progress.empty()
            return True
            
        except Exception as e:
            st.error(f"❌ ML training failed: {str(e)}")
            if 'progress' in locals():
                progress.empty()
            return False
    
    def detect_patterns_ml(self, df: pd.DataFrame, confidence_threshold: float = 0.6) -> Dict[str, List[Dict]]:
        """Detect patterns using ML models with comprehensive pattern analysis"""
        if not self.is_trained:
            if not self.train_models():
                return {}
        
        try:
            # Process data in sliding windows
            window_size = 50  # Same as synthetic data length
            features_list = []
            window_positions = []
            
            # Ensure we have enough data
            if len(df) < window_size + 10:
                return {}
            
            for i in range(window_size, len(df)):
                window_df = df.iloc[i-window_size:i]
                features = self._extract_pattern_features(window_df)
                if len(features) > 0:
                    features_list.append(features)
                    window_positions.append(i)
            
            if len(features_list) == 0:
                return {}
            
            # Convert to numpy array
            features = np.array(features_list)
            
            # Validate feature dimensions
            if features.shape[1] != 22:  # Expected number of features
                st.warning(f"Feature dimension mismatch: expected 22, got {features.shape[1]}")
                return {}
            
            # Get predictions
            try:
                rf_proba = self.rf_model.predict_proba(features)
                svm_proba = self.svm_model.predict_proba(self.scaler.transform(features))
                
                # Ensemble predictions
                ensemble_proba = (rf_proba * 0.6 + svm_proba * 0.4)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return {}
            
            # Process results with comprehensive pattern analysis
            pattern_names = ['Head and Shoulders', 'Double Bottom', 'Double Top', 'Cup and Handle']
            results = {name: [] for name in pattern_names}
            
            for i, proba in enumerate(ensemble_proba):
                max_confidence = np.max(proba)
                predicted_class = np.argmax(proba)
                
                if max_confidence >= confidence_threshold:
                    pattern_type = pattern_names[predicted_class]
                    position_idx = window_positions[i]  # Use the actual window position
                    
                    if position_idx < len(df):
                        current_price = float(df['Close'].iloc[position_idx])
                        current_volume = float(df['Volume'].iloc[position_idx]) if 'Volume' in df.columns else 0
                        
                        # Enhanced pattern analysis
                        pattern_analysis = self._analyze_ml_pattern_characteristics(
                            df, position_idx, window_size, pattern_type, predicted_class
                        )
                        
                        # Calculate target price using measured move
                        target_price = self._calculate_ml_target_price(
                            df, position_idx, pattern_type, predicted_class, pattern_analysis
                        )
                        
                        # Get technical indicators
                        technical_indicators = self._get_technical_indicators(df, position_idx)
                        
                        # Pattern duration and timing
                        pattern_duration = self._calculate_pattern_duration(df, position_idx, window_size)
                        
                        # Volume analysis
                        volume_analysis = self._analyze_volume_characteristics(df, position_idx, window_size)
                        
                        # Risk assessment
                        risk_assessment = self._assess_pattern_risk(df, position_idx, pattern_type, max_confidence)
                        
                        pattern_info = {
                            # Basic pattern information
                            'pattern_type': pattern_type,
                            'confidence': float(max_confidence),
                            'rf_confidence': float(rf_proba[i][predicted_class]),
                            'svm_confidence': float(svm_proba[i][predicted_class]),
                            'status': 'ml_detected',
                            'detection_method': 'ml_enhanced',
                            
                            # Price information
                            'current_price': current_price,
                            'target_price': float(target_price),
                            'price_change_potential': float((target_price - current_price) / current_price * 100),
                            'pattern_direction': 'bullish' if predicted_class in [1, 3] else 'bearish',
                            
                            # Timing and duration
                            'detection_date': df['Date'].iloc[position_idx] if 'Date' in df.columns else position_idx,
                            'pattern_duration_days': pattern_duration,
                            'position_idx': position_idx,
                            
                            # Volume analysis
                            'current_volume': current_volume,
                            'volume_trend': volume_analysis['volume_trend'],
                            'volume_confirmation': volume_analysis['volume_confirmation'],
                            'avg_volume': volume_analysis['avg_volume'],
                            
                            # Technical indicators
                            'rsi': technical_indicators['rsi'],
                            'macd': technical_indicators['macd'],
                            'bollinger_position': technical_indicators['bollinger_position'],
                            'moving_averages': technical_indicators['moving_averages'],
                            
                            # Pattern characteristics
                            'pattern_strength': pattern_analysis['pattern_strength'],
                            'symmetry_score': pattern_analysis['symmetry_score'],
                            'trend_alignment': pattern_analysis['trend_alignment'],
                            'support_resistance_levels': pattern_analysis['support_resistance_levels'],
                            
                            # Risk assessment
                            'risk_level': risk_assessment['risk_level'],
                            'stop_loss_suggestion': risk_assessment['stop_loss'],
                            'risk_reward_ratio': risk_assessment['risk_reward_ratio'],
                            'market_conditions': risk_assessment['market_conditions'],
                            
                            # Additional analysis
                            'pattern_completion': pattern_analysis['completion_percentage'],
                            'breakout_potential': pattern_analysis['breakout_potential'],
                            'false_signal_probability': 1 - max_confidence,
                            'recommended_action': self._get_pattern_recommendation(pattern_type, max_confidence, predicted_class)
                        }
                        
                        results[pattern_type].append(pattern_info)
            
            # Sort by confidence and limit results
            for pattern_type in results:
                results[pattern_type] = sorted(results[pattern_type], key=lambda x: x['confidence'], reverse=True)[:3]
            
            return results
            
        except Exception as e:
            st.error(f"ML detection error: {str(e)}")
            return {}
    
    def _analyze_ml_pattern_characteristics(self, df: pd.DataFrame, position_idx: int, window_size: int, 
                                          pattern_type: str, predicted_class: int) -> Dict:
        """Analyze pattern characteristics for ML-detected patterns"""
        try:
            # Extract pattern window
            start_idx = max(0, position_idx - window_size)
            end_idx = min(len(df), position_idx + 10)
            pattern_window = df.iloc[start_idx:end_idx]
            
            prices = pattern_window['Close'].values
            volumes = pattern_window['Volume'].values if 'Volume' in pattern_window.columns else None
            
            # Pattern strength analysis
            price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
            volatility = np.std(prices) / np.mean(prices)
            
            # Symmetry analysis
            mid_point = len(prices) // 2
            left_half = prices[:mid_point]
            right_half = prices[mid_point:]
            symmetry_score = 1 - abs(np.mean(left_half) - np.mean(right_half)) / np.mean(prices)
            
            # Trend alignment
            trend_slope = np.polyfit(np.arange(len(prices)), prices, 1)[0]
            trend_alignment = 'bullish' if trend_slope > 0 else 'bearish'
            
            # Support/Resistance levels
            peaks = argrelextrema(prices, np.greater, order=3)[0]
            troughs = argrelextrema(prices, np.less, order=3)[0]
            
            support_levels = [prices[i] for i in troughs] if len(troughs) > 0 else []
            resistance_levels = [prices[i] for i in peaks] if len(peaks) > 0 else []
            
            # Pattern completion percentage
            completion_percentage = self._estimate_pattern_completion(prices, pattern_type, predicted_class)
            
            # Breakout potential
            breakout_potential = self._assess_breakout_potential(prices, pattern_type, predicted_class)
            
            return {
                'pattern_strength': min(1.0, price_range * 2),
                'symmetry_score': max(0.1, min(1.0, symmetry_score)),
                'trend_alignment': trend_alignment,
                'support_resistance_levels': {
                    'support': support_levels[-3:] if len(support_levels) > 0 else [],
                    'resistance': resistance_levels[-3:] if len(resistance_levels) > 0 else []
                },
                'completion_percentage': completion_percentage,
                'breakout_potential': breakout_potential,
                'volatility': volatility
            }
            
        except Exception:
            return {
                'pattern_strength': 0.5,
                'symmetry_score': 0.5,
                'trend_alignment': 'neutral',
                'support_resistance_levels': {'support': [], 'resistance': []},
                'completion_percentage': 0.5,
                'breakout_potential': 0.5,
                'volatility': 0.02
            }
    
    def _calculate_ml_target_price(self, df: pd.DataFrame, position_idx: int, pattern_type: str, 
                                 predicted_class: int, pattern_analysis: Dict) -> float:
        """Calculate target price using measured move and pattern analysis"""
        try:
            current_price = df['Close'].iloc[position_idx]
            
            # Extract pattern window for measurement
            window_size = 50
            start_idx = max(0, position_idx - window_size)
            pattern_window = df.iloc[start_idx:position_idx]
            prices = pattern_window['Close'].values
            
            if len(prices) < 20:
                # Fallback to simple percentage-based targets
                if predicted_class in [1, 3]:  # Bullish patterns
                    return current_price * 1.15
                else:  # Bearish patterns
                    return current_price * 0.85
            
            # Calculate pattern height for measured move
            pattern_height = np.max(prices) - np.min(prices)
            
            if predicted_class in [1, 3]:  # Bullish patterns (Double Bottom, Cup and Handle)
                # Target = Current price + pattern height
                target_price = current_price + pattern_height
            else:  # Bearish patterns (Head and Shoulders, Double Top)
                # Target = Current price - pattern height
                target_price = current_price - pattern_height
            
            # Apply pattern-specific adjustments
            if pattern_type == 'Head and Shoulders':
                target_price *= 0.9  # Conservative target for H&S
            elif pattern_type == 'Double Bottom':
                target_price *= 1.1  # Optimistic target for double bottom
            elif pattern_type == 'Double Top':
                target_price *= 0.9  # Conservative target for double top
            elif pattern_type == 'Cup and Handle':
                target_price *= 1.05  # Moderate target for cup and handle
            
            return max(0.1, target_price)  # Ensure positive price
            
        except Exception:
            # Fallback calculation
            current_price = df['Close'].iloc[position_idx]
            if predicted_class in [1, 3]:  # Bullish patterns
                return current_price * 1.15
            else:  # Bearish patterns
                return current_price * 0.85
    
    def _get_technical_indicators(self, df: pd.DataFrame, position_idx: int) -> Dict:
        """Get technical indicators at the pattern position"""
        try:
            # RSI calculation
            window = 14
            start_idx = max(0, position_idx - window)
            prices = df['Close'].iloc[start_idx:position_idx+1].values
            
            if len(prices) >= 2:
                deltas = np.diff(prices)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
            else:
                rsi = 50
            
            # MACD calculation
            if len(prices) >= 26:
                ema12 = np.mean(prices[-12:]) if len(prices) >= 12 else prices[-1]
                ema26 = np.mean(prices[-26:]) if len(prices) >= 26 else prices[-1]
                macd = ema12 - ema26
            else:
                macd = 0
            
            # Bollinger Bands position
            if len(prices) >= 20:
                sma = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                current_price = prices[-1]
                bb_position = (current_price - (sma - 2*std)) / (4*std) if std > 0 else 0.5
            else:
                bb_position = 0.5
            
            # Moving averages
            ma5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
            ma20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            
            return {
                'rsi': float(rsi),
                'macd': float(macd),
                'bollinger_position': float(bb_position),
                'moving_averages': {
                    'ma5': float(ma5),
                    'ma20': float(ma20),
                    'ma5_above_ma20': ma5 > ma20
                }
            }
            
        except Exception:
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'bollinger_position': 0.5,
                'moving_averages': {
                    'ma5': 0.0,
                    'ma20': 0.0,
                    'ma5_above_ma20': True
                }
            }
    
    def _calculate_pattern_duration(self, df: pd.DataFrame, position_idx: int, window_size: int) -> int:
        """Calculate pattern duration in days"""
        try:
            start_idx = max(0, position_idx - window_size)
            if 'Date' in df.columns:
                start_date = df['Date'].iloc[start_idx]
                end_date = df['Date'].iloc[position_idx]
                duration = (end_date - start_date).days
            else:
                duration = window_size
            
            return max(1, duration)
            
        except Exception:
            return window_size
    
    def _analyze_volume_characteristics(self, df: pd.DataFrame, position_idx: int, window_size: int) -> Dict:
        """Analyze volume characteristics for pattern confirmation"""
        try:
            if 'Volume' not in df.columns:
                return {
                    'volume_trend': 'neutral',
                    'volume_confirmation': True,
                    'avg_volume': 0
                }
            
            start_idx = max(0, position_idx - window_size)
            volumes = df['Volume'].iloc[start_idx:position_idx+1].values
            
            if len(volumes) < 5:
                return {
                    'volume_trend': 'neutral',
                    'volume_confirmation': True,
                    'avg_volume': float(volumes[-1]) if len(volumes) > 0 else 0
                }
            
            # Volume trend
            recent_volumes = volumes[-5:]
            volume_trend_slope = np.polyfit(np.arange(len(recent_volumes)), recent_volumes, 1)[0]
            
            if volume_trend_slope > 0:
                volume_trend = 'increasing'
            elif volume_trend_slope < 0:
                volume_trend = 'decreasing'
            else:
                volume_trend = 'stable'
            
            # Volume confirmation
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes)
            volume_confirmation = current_volume > avg_volume * 0.8
            
            return {
                'volume_trend': volume_trend,
                'volume_confirmation': volume_confirmation,
                'avg_volume': float(avg_volume)
            }
            
        except Exception:
            return {
                'volume_trend': 'neutral',
                'volume_confirmation': True,
                'avg_volume': 0
            }
    
    def _assess_pattern_risk(self, df: pd.DataFrame, position_idx: int, pattern_type: str, confidence: float) -> Dict:
        """Assess risk level and provide risk management suggestions"""
        try:
            current_price = df['Close'].iloc[position_idx]
            
            # Calculate volatility-based risk
            window = 20
            start_idx = max(0, position_idx - window)
            prices = df['Close'].iloc[start_idx:position_idx+1].values
            volatility = np.std(prices) / np.mean(prices)
            
            # Risk level based on confidence and volatility
            if confidence > 0.8 and volatility < 0.02:
                risk_level = 'low'
            elif confidence > 0.6 and volatility < 0.04:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            # Stop loss suggestion
            if pattern_type in ['Head and Shoulders', 'Double Top']:  # Bearish patterns
                stop_loss = current_price * 1.05  # 5% above current price
            else:  # Bullish patterns
                stop_loss = current_price * 0.95  # 5% below current price
            
            # Risk-reward ratio
            target_price = self._calculate_ml_target_price(df, position_idx, pattern_type, 
                                                        0 if pattern_type in ['Head and Shoulders', 'Double Top'] else 1, {})
            potential_profit = abs(target_price - current_price)
            potential_loss = abs(stop_loss - current_price)
            risk_reward_ratio = potential_profit / potential_loss if potential_loss > 0 else 2.0
            
            # Market conditions assessment
            if len(prices) >= 20:
                trend = np.polyfit(np.arange(len(prices)), prices, 1)[0]
                if trend > 0:
                    market_conditions = 'bullish'
                elif trend < 0:
                    market_conditions = 'bearish'
                else:
                    market_conditions = 'sideways'
            else:
                market_conditions = 'neutral'
            
            return {
                'risk_level': risk_level,
                'stop_loss': float(stop_loss),
                'risk_reward_ratio': float(risk_reward_ratio),
                'market_conditions': market_conditions,
                'volatility': float(volatility)
            }
            
        except Exception:
            return {
                'risk_level': 'medium',
                'stop_loss': 0.0,
                'risk_reward_ratio': 1.5,
                'market_conditions': 'neutral',
                'volatility': 0.02
            }
    
    def _estimate_pattern_completion(self, prices: np.ndarray, pattern_type: str, predicted_class: int) -> float:
        """Estimate pattern completion percentage"""
        try:
            # Simple estimation based on price movement
            price_range = np.max(prices) - np.min(prices)
            current_range = prices[-1] - np.min(prices)
            
            if predicted_class in [1, 3]:  # Bullish patterns
                completion = min(1.0, current_range / price_range if price_range > 0 else 0.5)
            else:  # Bearish patterns
                completion = min(1.0, (np.max(prices) - prices[-1]) / price_range if price_range > 0 else 0.5)
            
            return max(0.1, min(1.0, completion))
            
        except Exception:
            return 0.5
    
    def _assess_breakout_potential(self, prices: np.ndarray, pattern_type: str, predicted_class: int) -> float:
        """Assess breakout potential based on pattern characteristics"""
        try:
            # Analyze recent price action
            recent_prices = prices[-10:] if len(prices) >= 10 else prices
            price_momentum = np.polyfit(np.arange(len(recent_prices)), recent_prices, 1)[0]
            
            # Volume trend would be better, but using price momentum as proxy
            if predicted_class in [1, 3]:  # Bullish patterns
                if price_momentum > 0:
                    return 0.8
                else:
                    return 0.4
            else:  # Bearish patterns
                if price_momentum < 0:
                    return 0.8
                else:
                    return 0.4
                    
        except Exception:
            return 0.5
    
    def _get_pattern_recommendation(self, pattern_type: str, confidence: float, predicted_class: int) -> str:
        """Get trading recommendation based on pattern and confidence"""
        try:
            if confidence < 0.5:
                return "Monitor - Low confidence pattern"
            elif confidence < 0.7:
                return "Watch - Medium confidence, wait for confirmation"
            else:
                if predicted_class in [1, 3]:  # Bullish patterns
                    return "Consider Buy - Strong bullish pattern detected"
                else:  # Bearish patterns
                    return "Consider Sell - Strong bearish pattern detected"
                    
        except Exception:
            return "Monitor - Pattern detected"

# =========================== ENHANCED VISUALIZATION ===========================

def create_enhanced_chart(df: pd.DataFrame, patterns: List[Dict], pattern_type: str, 
                         stock_name: str = "Stock", chart_type: str = "Candlestick") -> go.Figure:
    """Create enhanced chart with better pattern visualization"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{stock_name} - {pattern_type}', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    x_axis = df['Date'] if 'Date' in df.columns else df.index
    
    # Main price chart
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=x_axis,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26C281',
                decreasing_line_color='#ED5565',
                increasing_fillcolor='#26C281',
                decreasing_fillcolor='#ED5565'
            ),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#3498DB', width=2)
            ),
            row=1, col=1
        )
    
    # Add moving averages
    if len(df) >= 20:
        ma_20 = df['Close'].rolling(20).mean()
        ma_50 = df['Close'].rolling(50).mean() if len(df) >= 50 else None
        
        fig.add_trace(
            go.Scatter(x=x_axis, y=ma_20, mode='lines', name='MA20',
                      line=dict(color='orange', width=1, dash='dash')),
            row=1, col=1
        )
        
        if ma_50 is not None:
            fig.add_trace(
                go.Scatter(x=x_axis, y=ma_50, mode='lines', name='MA50',
                          line=dict(color='purple', width=1, dash='dot')),
                row=1, col=1
            )
    
    # Volume chart with color coding
    colors = ['#ED5565' if close < open else '#26C281' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=x_axis, y=df['Volume'], name='Volume', 
               marker_color=colors, opacity=0.6),
        row=2, col=1
    )
    
    # RSI
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss))
        
        fig.add_trace(
            go.Scatter(x=x_axis, y=rsi, mode='lines', name='RSI',
                      line=dict(color='#9B59B6', width=2)),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
    
    # Add pattern annotations
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    for i, pattern in enumerate(patterns[:3]):
        color = colors[i % len(colors)]
        confidence_text = f"{pattern['confidence']:.1%}"
        
        # Pattern-specific annotations
        if pattern_type == "Head and Shoulders" and pattern.get('detection_method') == 'traditional':
            _add_hs_annotations(fig, pattern, x_axis, color, confidence_text, i)
        elif pattern_type == "Double Bottom" and pattern.get('detection_method') == 'traditional':
            _add_db_annotations(fig, pattern, x_axis, color, confidence_text, i)
        elif pattern_type == "Double Top" and pattern.get('detection_method') == 'traditional':
            _add_dt_annotations(fig, pattern, x_axis, color, confidence_text, i)
        elif pattern_type == "Cup and Handle" and pattern.get('detection_method') == 'traditional':
            _add_ch_annotations(fig, pattern, x_axis, color, confidence_text, i)
        
        # Enhanced ML detection point
        if pattern.get('detection_method') in ['ml', 'ml_enhanced'] and 'position_idx' in pattern:
            fig.add_trace(
                go.Scatter(
                    x=[x_axis.iloc[pattern['position_idx']]],
                    y=[pattern['price']],
                    mode='markers+text',
                    name=f'ML Detection {i+1} ({confidence_text})',
                    marker=dict(size=20, color=color, symbol='diamond',
                              line=dict(width=3, color='white')),
                    text=[f'ML\n{pattern_type[:3]}'],
                    textposition='top center',
                    textfont=dict(size=10, color='white'),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add ML confidence indicator as a circle around the detection point
            fig.add_trace(
                go.Scatter(
                    x=[x_axis.iloc[pattern['position_idx']]],
                    y=[pattern['price']],
                    mode='markers',
                    name=f'ML Confidence {i+1}',
                    marker=dict(size=30, color=color, symbol='circle-open',
                              line=dict(width=3, color=color)),
                    opacity=pattern['confidence'],
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Enhanced target price line
        if 'target_price' in pattern:
            fig.add_hline(
                y=pattern['target_price'],
                line_dash="dot",
                line_color=color,
                line_width=3,
                annotation_text=f"🎯 Target: ${pattern['target_price']:.2f}",
                annotation_position="top right",
                annotation=dict(
                    font=dict(size=12, color=color),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=color,
                    borderwidth=1
                ),
                row=1, col=1,
                opacity=0.8
            )
            
            # Add target price markers at key pattern points
            if pattern.get('detection_method') == 'traditional':
                # Add target markers at pattern key points for better visibility
                if pattern_type == "Head and Shoulders" and 'head_idx' in pattern:
                    target_x = x_axis.iloc[pattern['head_idx']]
                elif pattern_type in ["Double Bottom", "Double Top"] and 'first_trough_idx' in pattern:
                    target_x = x_axis.iloc[pattern['first_trough_idx']]
                elif pattern_type == "Cup and Handle" and 'cup_bottom_idx' in pattern:
                    target_x = x_axis.iloc[pattern['cup_bottom_idx']]
                else:
                    target_x = x_axis.iloc[-1]  # Default to last point
                    
                fig.add_trace(
                    go.Scatter(
                        x=[target_x],
                        y=[pattern['target_price']],
                        mode='markers+text',
                        name=f'Target {i+1}',
                        marker=dict(size=12, color=color, symbol='star',
                                  line=dict(width=2, color='white')),
                        text=[f"${pattern['target_price']:.2f}"],
                        textposition='middle right',
                        textfont=dict(size=10, color=color),
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Layout updates
    fig.update_layout(
        title=dict(
            text=f"{stock_name} - {pattern_type} Pattern Analysis",
            x=0.5,
            font=dict(size=20, color='#2C3E50')
        ),
        height=800,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12, color='#2C3E50')
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#ECF0F1')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#ECF0F1')
    
    # RSI y-axis range
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    return fig

def _add_hs_annotations(fig, pattern, x_axis, color, confidence_text, index):
    """Add Head and Shoulders pattern annotations with clear labels"""
    if all(key in pattern for key in ['left_shoulder_idx', 'head_idx', 'right_shoulder_idx']):
        
        # Individual markers for each component with labels
        # Left Shoulder
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['left_shoulder_idx']]],
                y=[pattern['left_shoulder_price']],
                mode='markers+text',
                name=f'Left Shoulder {index+1}',
                marker=dict(size=15, color=color, symbol='triangle-up', 
                          line=dict(width=2, color='white')),
                text=['LS'],
                textposition='top center',
                textfont=dict(size=12, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Head
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['head_idx']]],
                y=[pattern['head_price']],
                mode='markers+text',
                name=f'Head {index+1}',
                marker=dict(size=18, color=color, symbol='triangle-up', 
                          line=dict(width=2, color='white')),
                text=['HEAD'],
                textposition='top center',
                textfont=dict(size=14, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Right Shoulder
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['right_shoulder_idx']]],
                y=[pattern['right_shoulder_price']],
                mode='markers+text',
                name=f'Right Shoulder {index+1}',
                marker=dict(size=15, color=color, symbol='triangle-up', 
                          line=dict(width=2, color='white')),
                text=['RS'],
                textposition='top center',
                textfont=dict(size=12, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Connect shoulders and head with lines
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['left_shoulder_idx']], 
                   x_axis.iloc[pattern['head_idx']], 
                   x_axis.iloc[pattern['right_shoulder_idx']]],
                y=[pattern['left_shoulder_price'], 
                   pattern['head_price'], 
                   pattern['right_shoulder_price']],
                mode='lines',
                name=f'H&S Pattern {index+1}',
                line=dict(color=color, width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Neckline with enhanced visibility
        if all(key in pattern for key in ['left_trough_idx', 'right_trough_idx']):
            neckline_level = pattern.get('neckline_level', 0)
            fig.add_trace(
                go.Scatter(
                    x=[x_axis.iloc[pattern['left_trough_idx']], 
                       x_axis.iloc[pattern['right_trough_idx']]],
                    y=[neckline_level, neckline_level],
                    mode='lines+text',
                    name=f'Neckline {index+1}',
                    line=dict(color=color, width=3, dash='dash'),
                    text=['', f'Neckline: ${neckline_level:.2f}'],
                    textposition='middle right',
                    textfont=dict(size=10, color=color),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Add neckline markers at the trough points
            fig.add_trace(
                go.Scatter(
                    x=[x_axis.iloc[pattern['left_trough_idx']], 
                       x_axis.iloc[pattern['right_trough_idx']]],
                    y=[neckline_level, neckline_level],
                    mode='markers',
                    name=f'Neckline Points {index+1}',
                    marker=dict(size=10, color=color, symbol='circle', 
                              line=dict(width=2, color='white')),
                    showlegend=False
                ),
                row=1, col=1
            )

def _add_db_annotations(fig, pattern, x_axis, color, confidence_text, index):
    """Add Double Bottom pattern annotations with clear labels"""
    if all(key in pattern for key in ['first_trough_idx', 'second_trough_idx']):
        
        # First Bottom
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['first_trough_idx']]],
                y=[pattern['first_trough_price']],
                mode='markers+text',
                name=f'First Bottom {index+1}',
                marker=dict(size=15, color=color, symbol='triangle-down', 
                          line=dict(width=2, color='white')),
                text=['B1'],
                textposition='bottom center',
                textfont=dict(size=12, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Second Bottom
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['second_trough_idx']]],
                y=[pattern['second_trough_price']],
                mode='markers+text',
                name=f'Second Bottom {index+1}',
                marker=dict(size=15, color=color, symbol='triangle-down', 
                          line=dict(width=2, color='white')),
                text=['B2'],
                textposition='bottom center',
                textfont=dict(size=12, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Connect the two bottoms
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['first_trough_idx']], 
                   x_axis.iloc[pattern['second_trough_idx']]],
                y=[pattern['first_trough_price'], 
                   pattern['second_trough_price']],
                mode='lines',
                name=f'Double Bottom {index+1}',
                line=dict(color=color, width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add resistance line if available
        if 'peak_between_idx' in pattern and 'peak_between_price' in pattern:
            fig.add_trace(
                go.Scatter(
                    x=[x_axis.iloc[pattern['first_trough_idx']], 
                       x_axis.iloc[pattern['peak_between_idx']],
                       x_axis.iloc[pattern['second_trough_idx']]],
                    y=[pattern['peak_between_price'], 
                       pattern['peak_between_price'],
                       pattern['peak_between_price']],
                    mode='lines+text',
                    name=f'Resistance Line {index+1}',
                    line=dict(color=color, width=2, dash='dash'),
                    text=['', f'Resistance: ${pattern["peak_between_price"]:.2f}', ''],
                    textposition='top center',
                    textfont=dict(size=10, color=color),
                    showlegend=True
                ),
                row=1, col=1
            )

def _add_dt_annotations(fig, pattern, x_axis, color, confidence_text, index):
    """Add Double Top pattern annotations with clear labels"""
    if all(key in pattern for key in ['first_peak_idx', 'second_peak_idx']):
        
        # First Peak
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['first_peak_idx']]],
                y=[pattern['first_peak_price']],
                mode='markers+text',
                name=f'First Peak {index+1}',
                marker=dict(size=15, color=color, symbol='triangle-up', 
                          line=dict(width=2, color='white')),
                text=['P1'],
                textposition='top center',
                textfont=dict(size=12, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Second Peak
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['second_peak_idx']]],
                y=[pattern['second_peak_price']],
                mode='markers+text',
                name=f'Second Peak {index+1}',
                marker=dict(size=15, color=color, symbol='triangle-up', 
                          line=dict(width=2, color='white')),
                text=['P2'],
                textposition='top center',
                textfont=dict(size=12, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Connect the two peaks
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['first_peak_idx']], 
                   x_axis.iloc[pattern['second_peak_idx']]],
                y=[pattern['first_peak_price'], 
                   pattern['second_peak_price']],
                mode='lines',
                name=f'Double Top {index+1}',
                line=dict(color=color, width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add support line if available
        if 'trough_between_idx' in pattern and 'trough_between_price' in pattern:
            fig.add_trace(
                go.Scatter(
                    x=[x_axis.iloc[pattern['first_peak_idx']], 
                       x_axis.iloc[pattern['trough_between_idx']],
                       x_axis.iloc[pattern['second_peak_idx']]],
                    y=[pattern['trough_between_price'], 
                       pattern['trough_between_price'],
                       pattern['trough_between_price']],
                    mode='lines+text',
                    name=f'Support Line {index+1}',
                    line=dict(color=color, width=2, dash='dash'),
                    text=['', f'Support: ${pattern["trough_between_price"]:.2f}', ''],
                    textposition='bottom center',
                    textfont=dict(size=10, color=color),
                    showlegend=True
                ),
                row=1, col=1
            )

def _add_ch_annotations(fig, pattern, x_axis, color, confidence_text, index):
    """Add Cup and Handle pattern annotations with clear labels"""
    if all(key in pattern for key in ['left_rim_idx', 'cup_bottom_idx', 'right_rim_idx', 'handle_bottom_idx']):
        
        # Left Rim
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['left_rim_idx']]],
                y=[pattern['left_rim_price']],
                mode='markers+text',
                name=f'Left Rim {index+1}',
                marker=dict(size=12, color=color, symbol='circle', 
                          line=dict(width=2, color='white')),
                text=['LR'],
                textposition='top center',
                textfont=dict(size=10, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Cup Bottom
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['cup_bottom_idx']]],
                y=[pattern['cup_bottom_price']],
                mode='markers+text',
                name=f'Cup Bottom {index+1}',
                marker=dict(size=15, color=color, symbol='circle', 
                          line=dict(width=2, color='white')),
                text=['CUP'],
                textposition='bottom center',
                textfont=dict(size=12, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Right Rim
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['right_rim_idx']]],
                y=[pattern['right_rim_price']],
                mode='markers+text',
                name=f'Right Rim {index+1}',
                marker=dict(size=12, color=color, symbol='circle', 
                          line=dict(width=2, color='white')),
                text=['RR'],
                textposition='top center',
                textfont=dict(size=10, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Handle Bottom
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['handle_bottom_idx']]],
                y=[pattern['handle_bottom_price']],
                mode='markers+text',
                name=f'Handle {index+1}',
                marker=dict(size=12, color=color, symbol='diamond', 
                          line=dict(width=2, color='white')),
                text=['HANDLE'],
                textposition='bottom center',
                textfont=dict(size=10, color='white'),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Draw the cup shape
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['left_rim_idx']], 
                   x_axis.iloc[pattern['cup_bottom_idx']], 
                   x_axis.iloc[pattern['right_rim_idx']]],
                y=[pattern['left_rim_price'], 
                   pattern['cup_bottom_price'], 
                   pattern['right_rim_price']],
                mode='lines',
                name=f'Cup Shape {index+1}',
                line=dict(color=color, width=2, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Draw the handle
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['right_rim_idx']], 
                   x_axis.iloc[pattern['handle_bottom_idx']]],
                y=[pattern['right_rim_price'], 
                   pattern['handle_bottom_price']],
                mode='lines',
                name=f'Handle Shape {index+1}',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add resistance line at rim level
        rim_level = max(pattern['left_rim_price'], pattern['right_rim_price'])
        fig.add_trace(
            go.Scatter(
                x=[x_axis.iloc[pattern['left_rim_idx']], 
                   x_axis.iloc[pattern['right_rim_idx']]],
                y=[rim_level, rim_level],
                mode='lines+text',
                name=f'Resistance Level {index+1}',
                line=dict(color=color, width=2, dash='dash'),
                text=['', f'Breakout: ${rim_level:.2f}'],
                textposition='top right',
                textfont=dict(size=10, color=color),
                showlegend=True
            ),
            row=1, col=1
        )

# =========================== STREAMLIT UI IMPROVEMENTS ===========================

def initialize_session_state():
    """Initialize session state with better defaults"""
    default_states = {
        'scan_results': [],
        'scan_completed': False,
        'selected_stock': None,
        'selected_pattern': None,
        'chart_type': "Candlestick",
        'ml_detector': None,
        'pattern_detector': PatternDetector(),
        'last_scan_time': None,
        'scan_settings': {}
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_ml_detector():
    """Get ML detector instance"""
    return ImprovedMLDetector()

def create_sidebar():
    """Create enhanced sidebar with better organization"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Date range
        with st.expander("📅 Date Range", expanded=True):
            default_start = datetime.date.today() - datetime.timedelta(days=DEFAULT_SCAN_DAYS)
            start_date = st.date_input("Start Date", value=default_start)
            end_date = st.date_input("End Date", value=datetime.date.today())
            
            if start_date >= end_date:
                st.error("Start date must be before end date")
                return None, None, None, None, None, None
        
        # Stock selection
        with st.expander("📊 Stock Selection", expanded=True):
            stock_input_method = st.radio(
                "Input Method:",
                ["Manual Entry", "Upload File", "Predefined Lists"],
                help="Choose how to specify stocks to scan"
            )
            
            stock_symbols = get_stock_symbols(stock_input_method)
            
            if not stock_symbols:
                st.warning("Please specify at least one stock symbol")
                return None, None, None, None, None, None
            
            st.info(f"Selected {len(stock_symbols)} stocks")
        
        # Pattern selection
        with st.expander("🔍 Pattern Selection", expanded=True):
            pattern_options = {
                "📉 Head & Shoulders": "Head and Shoulders",
                "📈 Double Bottom": "Double Bottom", 
                "📉 Double Top": "Double Top",
                "☕ Cup & Handle": "Cup and Handle"
            }
            
            selected_patterns = []
            for display_name, pattern_name in pattern_options.items():
                if st.checkbox(display_name, value=True):
                    selected_patterns.append(pattern_name)
            
            if not selected_patterns:
                st.warning("Please select at least one pattern")
                return None, None, None, None, None, None
        
        # Detection settings
        with st.expander("🎯 Detection Settings", expanded=False):
            use_traditional = st.checkbox("📊 Traditional Detection", value=True)
            use_ml = st.checkbox("🤖 ML Detection", value=False)
            
            confidence_threshold = st.slider(
                "Minimum Confidence",
                min_value=0.3, max_value=0.9, value=0.6, step=0.05,
                help="Minimum confidence level for pattern detection"
            )
            
            show_details = st.checkbox("Show Advanced Details", value=False)
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            max_patterns_per_stock = st.slider(
                "Max Patterns per Stock", 1, 10, 3,
                help="Maximum number of patterns to show per stock"
            )
            
            min_pattern_duration = st.slider(
                "Min Pattern Duration (days)", 20, 120, 30,
                help="Minimum duration for valid patterns"
            )
        
        return (start_date, end_date, stock_symbols, selected_patterns, 
                use_traditional, use_ml, confidence_threshold, show_details)

def get_stock_symbols(input_method):
    """Get stock symbols based on input method"""
    if input_method == "Manual Entry":
        stock_input = st.text_area(
            "Enter stock symbols (one per line):",
            value="AAPL\nMSFT\nGOOGL\nTSLA\nAMZN",
            height=100,
            help="Enter stock symbols separated by new lines"
        )
        return [s.strip().upper() for s in stock_input.split('\n') if s.strip()]
    
    elif input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload CSV/TXT file with stock symbols", 
            type=['csv', 'txt'],
            help="Upload a file containing stock symbols"
        )
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            return [s.strip().upper() for s in content.replace(',', '\n').split('\n') if s.strip()]
        else:
            return ['AAPL', 'MSFT', 'GOOGL']
    
    else:  # Predefined Lists
        list_options = {
            "🏛️ Blue Chip Stocks": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            "💻 Tech Giants": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'CRM', 'ORCL'],
            "🏦 Financial Sector": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BK'],
            "🏥 Healthcare": ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'ABT', 'MRK', 'BMY'],
            "🏭 Industrial": ['GE', 'CAT', 'BA', 'MMM', 'HON', 'UPS', 'LMT', 'RTX'],
            "🛒 Consumer": ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT', 'KO']
        }
        
        selected_list = st.selectbox("Choose predefined list:", list(list_options.keys()))
        return list_options[selected_list]

def run_pattern_scan(start_date, end_date, stock_symbols, selected_patterns, 
                    use_traditional, use_ml, confidence_threshold):
    """Run the pattern scanning process with progress tracking"""
    
    # Initialize detectors
    pattern_detector = st.session_state.pattern_detector
    ml_detector = None
    
    if use_ml:
        ml_detector = get_ml_detector()
        if not ml_detector.is_trained:
            if not ml_detector.train_models():
                st.error("Failed to initialize ML models")
                return []
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.info(f"🔍 Scanning {len(stock_symbols)} stocks for patterns...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        failed_symbols = []
        processed = 0
        
        for i, symbol in enumerate(stock_symbols):
            try:
                status_text.text(f"Processing {symbol}... ({i+1}/{len(stock_symbols)})")
                progress_bar.progress((i + 1) / len(stock_symbols))
                
                # Fetch data
                df = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if df is None or not validate_dataframe(df):
                    failed_symbols.append(symbol)
                    continue
                
                # Detect patterns
                all_patterns = {}
                
                # Traditional detection
                if use_traditional:
                    for pattern_name in selected_patterns:
                        if pattern_name == "Head and Shoulders":
                            patterns = pattern_detector.detect_head_and_shoulders(df)
                        elif pattern_name == "Double Bottom":
                            patterns = pattern_detector.detect_double_bottom(df)
                        elif pattern_name == "Double Top":
                            patterns = pattern_detector.detect_double_top(df)
                        elif pattern_name == "Cup and Handle":
                            patterns = pattern_detector.detect_cup_and_handle(df)
                        else:
                            patterns = []
                        
                        # Filter by confidence
                        patterns = [p for p in patterns if p['confidence'] >= confidence_threshold]
                        all_patterns[pattern_name] = patterns
                
                # ML detection
                if use_ml and ml_detector:
                    try:
                        ml_patterns = ml_detector.detect_patterns_ml(df, confidence_threshold)
                        # Merge ML results
                        for pattern_name in selected_patterns:
                            if pattern_name in ml_patterns:
                                if pattern_name not in all_patterns:
                                    all_patterns[pattern_name] = []
                                all_patterns[pattern_name].extend(ml_patterns[pattern_name])
                    except Exception as e:
                        st.warning(f"ML detection failed for {symbol}: {str(e)}")
                
                # Store results
                if any(len(patterns) > 0 for patterns in all_patterns.values()):
                    try:
                        current_price = float(df['Close'].iloc[-1])
                        volume = int(df['Volume'].iloc[-1])
                    except:
                        current_price = None
                        volume = None
                    
                    stock_result = {
                        'Symbol': symbol,
                        'Current_Price': current_price,
                        'Volume': volume,
                        'Patterns': all_patterns,
                        'Data': df,
                        'Last_Update': df['Date'].iloc[-1] if 'Date' in df.columns else None
                    }
                    results.append(stock_result)
                
                processed += 1
                
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
                failed_symbols.append(symbol)
        
        progress_bar.empty()
        status_text.empty()
    
    # Summary
    if results:
        total_patterns = sum(len(patterns) for stock in results for patterns in stock['Patterns'].values())
        st.success(f"✅ Scan completed! Found {total_patterns} patterns across {len(results)} stocks.")
    else:
        st.warning("No patterns found in the selected stocks.")
    
    if failed_symbols:
        st.warning(f"⚠️ Failed to process: {', '.join(failed_symbols[:5])}" + 
                  (f" and {len(failed_symbols)-5} more" if len(failed_symbols) > 5 else ""))
    
    return results

def display_scan_results():
    """Display enhanced scan results"""
    results = st.session_state.scan_results
    
    if not results:
        st.info("No scan results to display. Run a scan to see patterns.")
        return
    
    # Summary metrics
    display_summary_metrics(results)
    
    # Results table
    display_results_table(results)
    
    # Pattern analysis
    display_pattern_analysis(results)

def display_summary_metrics(results):
    """Display summary metrics with enhanced styling"""
    total_patterns = sum(len(patterns) for stock in results for patterns in stock['Patterns'].values())
    pattern_types = set()
    confirmed_patterns = 0
    
    for stock in results:
        for pattern_name, patterns in stock['Patterns'].items():
            if patterns:
                pattern_types.add(pattern_name)
                confirmed_patterns += sum(1 for p in patterns if p.get('status') == 'confirmed')
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 Stocks Scanned", 
            len(results),
            help="Total number of stocks analyzed"
        )
    
    with col2:
        st.metric(
            "🎯 Total Patterns", 
            total_patterns,
            help="Total patterns detected across all stocks"
        )
    
    with col3:
        st.metric(
            "✅ Confirmed", 
            confirmed_patterns,
            delta=f"{confirmed_patterns/total_patterns:.1%} of total" if total_patterns > 0 else "0%",
            help="Patterns with confirmed breakouts"
        )
    
    with col4:
        st.metric(
            "📈 Pattern Types", 
            len(pattern_types),
            help="Number of different pattern types found"
        )

def display_results_table(results):
    """Display enhanced results table with comprehensive pattern information"""
    st.subheader("📋 Detailed Results")
    
    # Create summary dataframe
    summary_data = []
    for stock in results:
        row = {
            'Symbol': stock['Symbol'],
            'Price': f"${stock['Current_Price']:.2f}" if stock['Current_Price'] else "N/A",
            'Volume': f"{stock['Volume']:,}" if stock['Volume'] else "N/A"
        }
        
        # Add pattern counts
        total_patterns = 0
        for pattern_name, patterns in stock['Patterns'].items():
            count = len(patterns)
            if count > 0:
                icon = PATTERN_DISPLAY[pattern_name]['icon']
                row[f"{icon} {pattern_name}"] = count
                total_patterns += count
        
        row['Total Patterns'] = total_patterns
        
        # Add last update
        if stock.get('Last_Update'):
            row['Last Update'] = stock['Last_Update'].strftime('%Y-%m-%d')
        
        summary_data.append(row)
    
    # Sort by total patterns
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Total Patterns', ascending=False)
    
    # Display with styling
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Stock", width="small"),
            "Price": st.column_config.TextColumn("Current Price", width="small"),
            "Volume": st.column_config.TextColumn("Volume", width="medium"),
            "Total Patterns": st.column_config.NumberColumn("Total", width="small")
        }
    )
    
    # Display comprehensive pattern details
    display_comprehensive_pattern_details(results)

def display_comprehensive_pattern_details(results):
    """Display comprehensive pattern details with all ML analysis information"""
    st.subheader("🤖 ML Pattern Analysis Details")
    
    # Filter stocks with ML patterns
    stocks_with_ml_patterns = []
    for stock in results:
        for pattern_name, patterns in stock['Patterns'].items():
            for pattern in patterns:
                if pattern.get('detection_method') == 'ml_enhanced':
                    stocks_with_ml_patterns.append({
                        'stock': stock,
                        'pattern_name': pattern_name,
                        'pattern': pattern
                    })
    
    if not stocks_with_ml_patterns:
        st.info("No ML-detected patterns found. Run ML detection to see comprehensive analysis.")
        return
    
    # Group by stock
    for stock in results:
        stock_ml_patterns = [item for item in stocks_with_ml_patterns if item['stock']['Symbol'] == stock['Symbol']]
        
        if stock_ml_patterns:
            st.markdown(f"### 📊 {stock['Symbol']} - ML Pattern Analysis")
            
            for item in stock_ml_patterns:
                pattern = item['pattern']
                pattern_name = item['pattern_name']
                
                # Create expandable section for each pattern
                with st.expander(f"🔍 {pattern_name} - Confidence: {pattern['confidence']:.1%}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 📈 Pattern Information")
                        st.write(f"**Pattern Type:** {pattern['pattern_type']}")
                        st.write(f"**Direction:** {pattern['pattern_direction']}")
                        st.write(f"**Confidence:** {pattern['confidence']:.1%}")
                        st.write(f"**RF Confidence:** {pattern['rf_confidence']:.1%}")
                        st.write(f"**SVM Confidence:** {pattern['svm_confidence']:.1%}")
                        st.write(f"**Status:** {pattern['status']}")
                        st.write(f"**Detection Date:** {pattern['detection_date']}")
                        st.write(f"**Duration:** {pattern['pattern_duration_days']} days")
                        
                        st.markdown("#### 💰 Price Analysis")
                        st.write(f"**Current Price:** ${pattern['current_price']:.2f}")
                        st.write(f"**Target Price:** ${pattern['target_price']:.2f}")
                        st.write(f"**Potential Change:** {pattern['price_change_potential']:+.1f}%")
                        
                        st.markdown("#### 📊 Technical Indicators")
                        st.write(f"**RSI:** {pattern['rsi']:.1f}")
                        st.write(f"**MACD:** {pattern['macd']:.3f}")
                        st.write(f"**Bollinger Position:** {pattern['bollinger_position']:.2f}")
                        st.write(f"**MA5 > MA20:** {pattern['moving_averages']['ma5_above_ma20']}")
                    
                    with col2:
                        st.markdown("#### 📊 Volume Analysis")
                        st.write(f"**Current Volume:** {pattern['current_volume']:,.0f}")
                        st.write(f"**Volume Trend:** {pattern['volume_trend']}")
                        st.write(f"**Volume Confirmation:** {pattern['volume_confirmation']}")
                        st.write(f"**Average Volume:** {pattern['avg_volume']:,.0f}")
                        
                        st.markdown("#### 🎯 Pattern Characteristics")
                        st.write(f"**Pattern Strength:** {pattern['pattern_strength']:.1%}")
                        st.write(f"**Symmetry Score:** {pattern['symmetry_score']:.1%}")
                        st.write(f"**Trend Alignment:** {pattern['trend_alignment']}")
                        st.write(f"**Completion:** {pattern['pattern_completion']:.1%}")
                        st.write(f"**Breakout Potential:** {pattern['breakout_potential']:.1%}")
                        
                        st.markdown("#### ⚠️ Risk Assessment")
                        st.write(f"**Risk Level:** {pattern['risk_level'].title()}")
                        st.write(f"**Stop Loss:** ${pattern['stop_loss_suggestion']:.2f}")
                        st.write(f"**Risk/Reward Ratio:** {pattern['risk_reward_ratio']:.2f}")
                        st.write(f"**Market Conditions:** {pattern['market_conditions']}")
                        st.write(f"**False Signal Probability:** {pattern['false_signal_probability']:.1%}")
                    
                    # Recommendation section
                    st.markdown("#### 💡 Trading Recommendation")
                    recommendation_color = "green" if "Buy" in pattern['recommended_action'] else "red" if "Sell" in pattern['recommended_action'] else "orange"
                    st.markdown(f"<div style='background-color: {recommendation_color}20; padding: 10px; border-radius: 5px; border-left: 4px solid {recommendation_color};'>"
                              f"<strong>{pattern['recommended_action']}</strong></div>", unsafe_allow_html=True)
                    
                    # Support/Resistance levels if available
                    if hasattr(pattern, 'support_resistance_levels'):
                        st.markdown("#### 🎯 Support/Resistance Levels")
                        support_levels = pattern.get('support_resistance_levels', {}).get('support', [])
                        resistance_levels = pattern.get('support_resistance_levels', {}).get('resistance', [])
                        
                        if support_levels:
                            st.write(f"**Support Levels:** {', '.join([f'${level:.2f}' for level in support_levels])}")
                        if resistance_levels:
                            st.write(f"**Resistance Levels:** {', '.join([f'${level:.2f}' for level in resistance_levels])}")
                    
                    st.markdown("---")

def display_pattern_analysis(results):
    """Display enhanced pattern analysis section"""
    st.subheader("🔍 Pattern Analysis")
    
    # Filter stocks with patterns
    stocks_with_patterns = [stock for stock in results 
                          if any(len(patterns) > 0 for patterns in stock['Patterns'].values())]
    
    if not stocks_with_patterns:
        st.info("No patterns found in the scanned stocks.")
        return
    
    # Create selection interface
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        stock_options = [stock['Symbol'] for stock in stocks_with_patterns]
        selected_stock = st.selectbox(
            "📊 Select Stock:",
            options=stock_options,
            key="stock_selector"
        )
    
    # Find selected stock data
    selected_data = next(stock for stock in stocks_with_patterns if stock['Symbol'] == selected_stock)
    
    with col2:
        # Get pattern options for selected stock
        pattern_options = []
        for pattern_name, patterns in selected_data["Patterns"].items():
            if patterns:
                icon = PATTERN_DISPLAY[pattern_name]['icon']
                pattern_options.append(f"{icon} {pattern_name} ({len(patterns)})")
        
        if pattern_options:
            selected_pattern_display = st.selectbox(
                "🎯 Select Pattern:",
                options=pattern_options,
                key="pattern_selector"
            )
            selected_pattern = selected_pattern_display.split(' (')[0].replace('📉 ', '').replace('📈 ', '').replace('☕ ', '').replace('📐 ', '')
        else:
            st.info("No patterns found for this stock.")
            return
    
    with col3:
        chart_type = st.selectbox(
            "📈 Chart Type:",
            options=["Candlestick", "Line", "OHLC"],
            key="chart_type_selector"
        )
    
    # Display pattern analysis
    if selected_pattern:
        display_pattern_details_enhanced(selected_data, selected_pattern, chart_type)

def display_pattern_details_enhanced(selected_data, selected_pattern, chart_type):
    """Display enhanced pattern details with better visualization"""
    patterns = selected_data["Patterns"][selected_pattern]
    
    if not patterns:
        st.info("No patterns of this type found.")
        return
    
    # Pattern summary
    st.subheader(f"📊 {selected_data['Symbol']} - {selected_pattern}")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Patterns Found", len(patterns))
    
    with col2:
        avg_confidence = sum(p['confidence'] for p in patterns) / len(patterns)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        if patterns and 'duration_days' in patterns[0]:
            avg_duration = sum(p.get('duration_days', 0) for p in patterns) / len(patterns)
            st.metric("Avg Duration", f"{avg_duration:.0f} days")
        else:
            detection_methods = [p.get('detection_method', 'unknown') for p in patterns]
            most_common = max(set(detection_methods), key=detection_methods.count)
            st.metric("Detection Method", most_common.title())
    
    with col4:
        confirmed = sum(1 for p in patterns if p.get('status') == 'confirmed')
        st.metric("Status", f"{confirmed} Confirmed")
    
    # Chart visualization
    fig = create_enhanced_chart(
        selected_data["Data"],
        patterns,
        selected_pattern,
        stock_name=selected_data["Symbol"],
        chart_type=chart_type
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Pattern details
    display_individual_patterns(selected_data, selected_pattern, patterns)

def display_individual_patterns(selected_data, selected_pattern, patterns):
    """Display individual pattern details"""
    st.subheader("📋 Pattern Details")
    
    for i, pattern in enumerate(patterns[:3]):  # Show top 3 patterns
        with st.expander(
            f"🎯 Pattern #{i+1} - Confidence: {pattern['confidence']:.1%} - "
            f"Status: {pattern.get('status', 'N/A').title()}", 
            expanded=(i == 0)
        ):
            # Create tabs for different aspects
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "📅 Timeline", "💹 Trading Info"])
            
            with tab1:
                display_pattern_overview(pattern, selected_pattern)
            
            with tab2:
                display_pattern_timeline(selected_data, pattern, selected_pattern)
            
            with tab3:
                display_trading_info(selected_data, pattern)

def display_pattern_overview(pattern, selected_pattern):
    """Display pattern overview information"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Pattern Metrics**")
        st.write(f"• Confidence: {pattern['confidence']:.1%}")
        st.write(f"• Status: {pattern.get('status', 'N/A').title()}")
        st.write(f"• Detection: {pattern.get('detection_method', 'N/A').title()}")
        if 'duration_days' in pattern:
            st.write(f"• Duration: {pattern['duration_days']} days")
    
    with col2:
        st.markdown("**💰 Price Targets**")
        current_price = pattern.get('price', 0)
        target_price = pattern.get('target_price', 0)
        
        if current_price and target_price:
            price_change = (target_price - current_price) / current_price * 100
            st.write(f"• Current: ${current_price:.2f}")
            st.write(f"• Target: ${target_price:.2f}")
            st.write(f"• Potential: {price_change:+.1f}%")
        
        if 'pattern_height' in pattern:
            st.write(f"• Pattern Height: ${pattern['pattern_height']:.2f}")

def display_pattern_timeline(selected_data, pattern, selected_pattern):
    """Display pattern timeline information"""
    try:
        dates_info = []
        
        if selected_pattern == "Head and Shoulders":
            if 'left_shoulder_idx' in pattern:
                date = selected_data["Data"]['Date'].iloc[pattern['left_shoulder_idx']]
                price = pattern.get('left_shoulder_price', 0)
                dates_info.append(("Left Shoulder", date, price))
            
            if 'head_idx' in pattern:
                date = selected_data["Data"]['Date'].iloc[pattern['head_idx']]
                price = pattern.get('head_price', 0)
                dates_info.append(("Head", date, price))
            
            if 'right_shoulder_idx' in pattern:
                date = selected_data["Data"]['Date'].iloc[pattern['right_shoulder_idx']]
                price = pattern.get('right_shoulder_price', 0)
                dates_info.append(("Right Shoulder", date, price))
        
        elif selected_pattern in ["Double Bottom", "Double Top"]:
            key_prefix = "trough" if "Bottom" in selected_pattern else "peak"
            price_key = f"{key_prefix}_price"
            
            if f'first_{key_prefix}_idx' in pattern:
                date = selected_data["Data"]['Date'].iloc[pattern[f'first_{key_prefix}_idx']]
                price = pattern.get(f'first_{price_key}', 0)
                dates_info.append((f"First {key_prefix.title()}", date, price))
            
            if f'second_{key_prefix}_idx' in pattern:
                date = selected_data["Data"]['Date'].iloc[pattern[f'second_{key_prefix}_idx']]
                price = pattern.get(f'second_{price_key}', 0)
                dates_info.append((f"Second {key_prefix.title()}", date, price))
        
        elif selected_pattern == "Cup and Handle":
            if 'left_rim_idx' in pattern:
                date = selected_data["Data"]['Date'].iloc[pattern['left_rim_idx']]
                price = pattern.get('left_rim_price', 0)
                dates_info.append(("Left Rim", date, price))
            
            if 'cup_bottom_idx' in pattern:
                date = selected_data["Data"]['Date'].iloc[pattern['cup_bottom_idx']]
                price = pattern.get('cup_bottom_price', 0)
                dates_info.append(("Cup Bottom", date, price))
        
        # Display timeline
        if dates_info:
            timeline_df = pd.DataFrame(dates_info, columns=['Event', 'Date', 'Price'])
            timeline_df['Price'] = timeline_df['Price'].apply(lambda x: f"${x:.2f}")
            timeline_df['Date'] = timeline_df['Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)
        else:
            st.info("Timeline information not available for this pattern.")
            
    except Exception as e:
        st.warning(f"Could not display timeline: {str(e)}")

def display_trading_info(selected_data, pattern):
    """Display trading-related information"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Trading Signals**")
        
        # Pattern type and bias
        pattern_type = pattern.get('type', '')
        if pattern_type in PATTERN_DISPLAY:
            bias = PATTERN_DISPLAY[pattern_type]['type']
            st.write(f"• Market Bias: {bias.title()}")
        
        # Breakout status
        status = pattern.get('status', 'forming')
        if status == 'confirmed':
            st.write("• ✅ Breakout Confirmed")
        else:
            st.write("• ⏳ Pattern Forming")
    
    with col2:
        st.markdown("**⚠️ Risk Management**")
        
        if 'neckline_level' in pattern or 'neckline_price' in pattern:
            neckline = pattern.get('neckline_level', pattern.get('neckline_price', 0))
            st.write(f"• Stop Loss: ${neckline:.2f}")
        
        confidence = pattern.get('confidence', 0)
        if confidence > 0.8:
            st.write("• Risk Level: Low")
        elif confidence > 0.6:
            st.write("• Risk Level: Medium")
        else:
            st.write("• Risk Level: High")

def main():
    """Enhanced main application"""
    st.set_page_config(
        page_title="Advanced Stock Pattern Scanner",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📈 Advanced Stock Pattern Scanner")
    st.markdown(
        "Detect chart patterns using traditional algorithms and machine learning. "
        "Find Head & Shoulders, Double Tops/Bottoms, Cup & Handle patterns and more."
    )
    
    # Sidebar configuration
    sidebar_result = create_sidebar()
    if sidebar_result[0] is None:  # Error in sidebar
        return
    
    (start_date, end_date, stock_symbols, selected_patterns, 
     use_traditional, use_ml, confidence_threshold, show_details) = sidebar_result
    
    # Main action buttons
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        scan_button = st.button(
            "🚀 Start Pattern Scan", 
            type="primary", 
            use_container_width=True,
            help="Begin scanning selected stocks for patterns"
        )
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.scan_results = []
            st.session_state.scan_completed = False
            st.rerun()
    
    with col3:
        export_button = st.button(
            "📊 Export Results", 
            use_container_width=True,
            disabled=not st.session_state.scan_results,
            help="Export scan results to CSV"
        )
    
    # Handle export
    if export_button and st.session_state.scan_results:
        export_results_to_csv()
    
    # Handle scan
    if scan_button:
        if not use_traditional and not use_ml:
            st.error("Please select at least one detection method.")
            return
        
        # Store scan settings
        st.session_state.scan_settings = {
            'start_date': start_date,
            'end_date': end_date,
            'symbols': stock_symbols,
            'patterns': selected_patterns,
            'traditional': use_traditional,
            'ml': use_ml,
            'confidence': confidence_threshold
        }
        
        # Run scan
        with st.spinner("🔍 Scanning stocks for patterns..."):
            results = run_pattern_scan(
                start_date, end_date, stock_symbols, selected_patterns,
                use_traditional, use_ml, confidence_threshold
            )
        
        if results:
            st.session_state.scan_results = results
            st.session_state.scan_completed = True
            st.session_state.last_scan_time = datetime.datetime.now()
            st.rerun()
    
    # Display results
    if st.session_state.scan_completed:
        st.markdown("---")
        display_scan_results()

def export_results_to_csv():
    """Export scan results to CSV format"""
    try:
        results = st.session_state.scan_results
        export_data = []
        
        for stock in results:
            for pattern_name, patterns in stock['Patterns'].items():
                for pattern in patterns:
                    export_data.append({
                        'Symbol': stock['Symbol'],
                        'Pattern_Type': pattern_name,
                        'Confidence': pattern['confidence'],
                        'Status': pattern.get('status', 'N/A'),
                        'Detection_Method': pattern.get('detection_method', 'N/A'),
                        'Target_Price': pattern.get('target_price', 'N/A'),
                        'Current_Price': stock.get('Current_Price', 'N/A'),
                        'Duration_Days': pattern.get('duration_days', 'N/A')
                    })
        
        if export_data:
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"pattern_scan_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
            st.success("✅ Results exported successfully!")
        else:
            st.warning("No data to export.")
            
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

if __name__ == "__main__":
    main()
                    