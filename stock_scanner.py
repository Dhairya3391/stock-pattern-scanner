"""
Stock Pattern Scanner - Consolidated Version
Advanced stock pattern detection with traditional algorithms and lightweight ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================== CONFIGURATION ===========================

# Default settings
DEFAULT_SCAN_DAYS = 365
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Pattern settings
PATTERN_SETTINGS = {
    'head_and_shoulders': {
        'shoulder_tolerance': 0.08,
        'neckline_tolerance': 0.05,
        'min_duration_days': 90,
        'breakout_threshold': 0.02
    },
    'double_bottom': {
        'price_tolerance': 0.03,
        'min_rebound': 0.05,
        'min_duration_days': 60,
        'breakout_threshold': 0.02
    },
    'double_top': {
        'price_tolerance': 0.03,
        'min_pullback': 0.05,
        'min_duration_days': 60,
        'breakdown_threshold': 0.02
    },
    'cup_and_handle': {
        'rim_tolerance': 0.05,
        'min_cup_depth': 0.15,
        'handle_retrace_min': 0.25,
        'handle_retrace_max': 0.50,
        'min_duration_days': 120,
        'breakout_threshold': 0.02
    }
}

PATTERN_DISPLAY = {
    'Head and Shoulders': {'icon': '📉', 'color': '#ff6b6b'},
    'Double Bottom': {'icon': '📈', 'color': '#51cf66'},
    'Double Top': {'icon': '📉', 'color': '#ff8787'},
    'Cup and Handle': {'icon': '☕', 'color': '#74c0fc'}
}

# =========================== DATA FETCHING ===========================

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_str, end=end_str)
        
        if df.empty or len(df) < 50:
            return None
            
        df = df.reset_index()
        
        # Ensure numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if df['Close'].isnull().all():
            return None
            
        return df
        
    except Exception as e:
        print(f"Error fetching {symbol}: {str(e)}")
        return None

# =========================== TRADITIONAL PATTERN DETECTION ===========================

def detect_head_and_shoulders(df, debug=False):
    """Simplified Head and Shoulders detection"""
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns or len(df) < 50:
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    troughs = argrelextrema(data['Close'].values, np.less, order=5)[0]
    
    if len(peaks) < 3:
        return []
    
    patterns = []
    
    for i in range(len(peaks) - 2):
        left_shoulder_idx = peaks[i]
        head_idx = peaks[i + 1] 
        right_shoulder_idx = peaks[i + 2]
        
        ls_price = data['Close'].iloc[left_shoulder_idx]
        head_price = data['Close'].iloc[head_idx]
        rs_price = data['Close'].iloc[right_shoulder_idx]
        
        if not (head_price > ls_price and head_price > rs_price):
            continue
            
        shoulder_diff = abs(ls_price - rs_price) / min(ls_price, rs_price)
        if shoulder_diff > PATTERN_SETTINGS['head_and_shoulders']['shoulder_tolerance']:
            continue
            
        left_troughs = [t for t in troughs if left_shoulder_idx < t < head_idx]
        right_troughs = [t for t in troughs if head_idx < t < right_shoulder_idx]
        
        if not left_troughs or not right_troughs:
            continue
            
        left_trough_idx = min(left_troughs, key=lambda x: data['Close'].iloc[x])
        right_trough_idx = min(right_troughs, key=lambda x: data['Close'].iloc[x])
        
        left_trough_price = data['Close'].iloc[left_trough_idx]
        right_trough_price = data['Close'].iloc[right_trough_idx]
        
        neckline_diff = abs(left_trough_price - right_trough_price) / min(left_trough_price, right_trough_price)
        if neckline_diff > PATTERN_SETTINGS['head_and_shoulders']['neckline_tolerance']:
            continue
            
        neckline_slope = (right_trough_price - left_trough_price) / (right_trough_idx - left_trough_idx)
        
        breakout_idx = None
        neckline_level = (left_trough_price + right_trough_price) / 2
        
        for j in range(right_shoulder_idx + 1, min(len(data), right_shoulder_idx + 30)):
            current_neckline = left_trough_price + neckline_slope * (j - left_trough_idx)
            if data['Close'].iloc[j] < current_neckline * (1 - PATTERN_SETTINGS['head_and_shoulders']['breakout_threshold']):
                breakout_idx = j
                break
        
        pattern_duration = (data['Date'].iloc[right_shoulder_idx] - data['Date'].iloc[left_shoulder_idx]).days
        if pattern_duration < PATTERN_SETTINGS['head_and_shoulders']['min_duration_days']:
            continue
            
        confidence = 0.7
        confidence += (1 - shoulder_diff / PATTERN_SETTINGS['head_and_shoulders']['shoulder_tolerance']) * 0.15
        confidence += (1 - neckline_diff / PATTERN_SETTINGS['head_and_shoulders']['neckline_tolerance']) * 0.10
        if breakout_idx:
            confidence += 0.05
            
        pattern_height = head_price - neckline_level
        target_price = neckline_level - pattern_height
        
        pattern = {
            'type': 'Head and Shoulders',
            'left_shoulder_idx': left_shoulder_idx,
            'head_idx': head_idx,
            'right_shoulder_idx': right_shoulder_idx,
            'left_trough_idx': left_trough_idx,
            'right_trough_idx': right_trough_idx,
            'breakout_idx': breakout_idx,
            'left_shoulder_price': float(ls_price),
            'head_price': float(head_price),
            'right_shoulder_price': float(rs_price),
            'neckline_level': float(neckline_level),
            'target_price': float(target_price),
            'pattern_height': float(pattern_height),
            'confidence': min(0.95, confidence),
            'status': 'confirmed' if breakout_idx else 'forming',
            'duration_days': pattern_duration,
            'detection_method': 'traditional'
        }
        
        patterns.append(pattern)
    
    return sorted(patterns, key=lambda x: x['confidence'], reverse=True)

def detect_double_bottom(df, debug=False):
    """Simplified Double Bottom detection"""
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns or len(df) < 50:
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    troughs = argrelextrema(data['Close'].values, np.less, order=5)[0]
    peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    
    if len(troughs) < 2:
        return []
    
    patterns = []
    
    for i in range(len(troughs) - 1):
        first_trough_idx = troughs[i]
        
        for j in range(i + 1, len(troughs)):
            second_trough_idx = troughs[j]
            
            first_price = data['Close'].iloc[first_trough_idx]
            second_price = data['Close'].iloc[second_trough_idx]
            
            price_diff = abs(first_price - second_price) / min(first_price, second_price)
            if price_diff > PATTERN_SETTINGS['double_bottom']['price_tolerance']:
                continue
                
            between_peaks = [p for p in peaks if first_trough_idx < p < second_trough_idx]
            if not between_peaks:
                continue
                
            neckline_idx = max(between_peaks, key=lambda x: data['Close'].iloc[x])
            neckline_price = data['Close'].iloc[neckline_idx]
            
            min_trough_price = min(first_price, second_price)
            rebound_strength = (neckline_price - min_trough_price) / min_trough_price
            if rebound_strength < PATTERN_SETTINGS['double_bottom']['min_rebound']:
                continue
                
            pattern_duration = (data['Date'].iloc[second_trough_idx] - data['Date'].iloc[first_trough_idx]).days
            if pattern_duration < PATTERN_SETTINGS['double_bottom']['min_duration_days']:
                continue
                
            breakout_idx = None
            for k in range(second_trough_idx + 1, min(len(data), second_trough_idx + 30)):
                if data['Close'].iloc[k] > neckline_price * (1 + PATTERN_SETTINGS['double_bottom']['breakout_threshold']):
                    breakout_idx = k
                    break
            
            confidence = 0.7
            confidence += (1 - price_diff / PATTERN_SETTINGS['double_bottom']['price_tolerance']) * 0.15
            confidence += min(0.10, rebound_strength * 2)
            if breakout_idx:
                confidence += 0.05
                
            pattern_height = neckline_price - min_trough_price
            target_price = neckline_price + pattern_height
            
            pattern = {
                'type': 'Double Bottom',
                'first_trough_idx': first_trough_idx,
                'second_trough_idx': second_trough_idx,
                'neckline_idx': neckline_idx,
                'breakout_idx': breakout_idx,
                'first_trough_price': float(first_price),
                'second_trough_price': float(second_price),
                'neckline_price': float(neckline_price),
                'target_price': float(target_price),
                'pattern_height': float(pattern_height),
                'confidence': min(0.95, confidence),
                'status': 'confirmed' if breakout_idx else 'forming',
                'duration_days': pattern_duration,
                'rebound_strength': rebound_strength,
                'detection_method': 'traditional'
            }
            
            patterns.append(pattern)
    
    return sorted(patterns, key=lambda x: x['confidence'], reverse=True)

def detect_double_top(df, debug=False):
    """Simplified Double Top detection"""
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns or len(df) < 50:
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
    troughs = argrelextrema(data['Close'].values, np.less, order=5)[0]
    
    if len(peaks) < 2:
        return []
    
    patterns = []
    
    for i in range(len(peaks) - 1):
        first_peak_idx = peaks[i]
        
        for j in range(i + 1, len(peaks)):
            second_peak_idx = peaks[j]
            
            first_price = data['Close'].iloc[first_peak_idx]
            second_price = data['Close'].iloc[second_peak_idx]
            
            price_diff = abs(first_price - second_price) / min(first_price, second_price)
            if price_diff > PATTERN_SETTINGS['double_top']['price_tolerance']:
                continue
                
            between_troughs = [t for t in troughs if first_peak_idx < t < second_peak_idx]
            if not between_troughs:
                continue
                
            neckline_idx = min(between_troughs, key=lambda x: data['Close'].iloc[x])
            neckline_price = data['Close'].iloc[neckline_idx]
            
            max_peak_price = max(first_price, second_price)
            pullback_strength = (max_peak_price - neckline_price) / max_peak_price
            if pullback_strength < PATTERN_SETTINGS['double_top']['min_pullback']:
                continue
                
            pattern_duration = (data['Date'].iloc[second_peak_idx] - data['Date'].iloc[first_peak_idx]).days
            if pattern_duration < PATTERN_SETTINGS['double_top']['min_duration_days']:
                continue
                
            breakdown_idx = None
            for k in range(second_peak_idx + 1, min(len(data), second_peak_idx + 30)):
                if data['Close'].iloc[k] < neckline_price * (1 - PATTERN_SETTINGS['double_top']['breakdown_threshold']):
                    breakdown_idx = k
                    break
            
            confidence = 0.7
            confidence += (1 - price_diff / PATTERN_SETTINGS['double_top']['price_tolerance']) * 0.15
            confidence += min(0.10, pullback_strength * 2)
            if breakdown_idx:
                confidence += 0.05
                
            pattern_height = max_peak_price - neckline_price
            target_price = neckline_price - pattern_height
            
            pattern = {
                'type': 'Double Top',
                'first_peak_idx': first_peak_idx,
                'second_peak_idx': second_peak_idx,
                'neckline_idx': neckline_idx,
                'breakdown_idx': breakdown_idx,
                'first_peak_price': float(first_price),
                'second_peak_price': float(second_price),
                'neckline_price': float(neckline_price),
                'target_price': float(target_price),
                'pattern_height': float(pattern_height),
                'confidence': min(0.95, confidence),
                'status': 'confirmed' if breakdown_idx else 'forming',
                'duration_days': pattern_duration,
                'pullback_strength': pullback_strength,
                'detection_method': 'traditional'
            }
            
            patterns.append(pattern)
    
    return sorted(patterns, key=lambda x: x['confidence'], reverse=True)

def detect_cup_and_handle(df, debug=False):
    """Simplified Cup and Handle detection"""
    if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns or len(df) < 100:
        return []
    
    data = df.copy()
    if 'Date' not in data.columns:
        data['Date'] = data.index
    
    peaks = argrelextrema(data['Close'].values, np.greater, order=8)[0]
    troughs = argrelextrema(data['Close'].values, np.less, order=8)[0]
    
    if len(peaks) < 2 or len(troughs) < 2:
        return []
    
    patterns = []
    
    for i in range(len(peaks) - 1):
        left_rim_idx = peaks[i]
        left_rim_price = data['Close'].iloc[left_rim_idx]
        
        cup_troughs = [t for t in troughs if left_rim_idx < t < left_rim_idx + 120]
        if not cup_troughs:
            continue
            
        cup_bottom_idx = min(cup_troughs, key=lambda x: data['Close'].iloc[x])
        cup_bottom_price = data['Close'].iloc[cup_bottom_idx]
        
        cup_depth = (left_rim_price - cup_bottom_price) / left_rim_price
        if cup_depth < PATTERN_SETTINGS['cup_and_handle']['min_cup_depth']:
            continue
            
        right_rim_candidates = [p for p in peaks if cup_bottom_idx < p < cup_bottom_idx + 80]
        if not right_rim_candidates:
            continue
            
        valid_right_rims = []
        for candidate in right_rim_candidates:
            candidate_price = data['Close'].iloc[candidate]
            rim_diff = abs(candidate_price - left_rim_price) / left_rim_price
            if rim_diff <= PATTERN_SETTINGS['cup_and_handle']['rim_tolerance']:
                valid_right_rims.append(candidate)
                
        if not valid_right_rims:
            continue
            
        right_rim_idx = valid_right_rims[0]
        right_rim_price = data['Close'].iloc[right_rim_idx]
        
        handle_troughs = [t for t in troughs if right_rim_idx < t < right_rim_idx + 30]
        if not handle_troughs:
            continue
            
        handle_bottom_idx = handle_troughs[0]
        handle_bottom_price = data['Close'].iloc[handle_bottom_idx]
        
        cup_height = left_rim_price - cup_bottom_price
        handle_retrace = (right_rim_price - handle_bottom_price) / cup_height
        
        if not (PATTERN_SETTINGS['cup_and_handle']['handle_retrace_min'] <= handle_retrace <= PATTERN_SETTINGS['cup_and_handle']['handle_retrace_max']):
            continue
            
        pattern_duration = (data['Date'].iloc[handle_bottom_idx] - data['Date'].iloc[left_rim_idx]).days
        if pattern_duration < PATTERN_SETTINGS['cup_and_handle']['min_duration_days']:
            continue
            
        breakout_idx = None
        rim_level = max(left_rim_price, right_rim_price)
        
        for k in range(handle_bottom_idx + 1, min(len(data), handle_bottom_idx + 20)):
            if data['Close'].iloc[k] > rim_level * (1 + PATTERN_SETTINGS['cup_and_handle']['breakout_threshold']):
                breakout_idx = k
                break
        
        confidence = 0.6
        confidence += min(0.15, cup_depth * 0.5)
        confidence += (1 - abs(handle_retrace - 0.375) / 0.125) * 0.10
        confidence += (1 - abs(left_rim_price - right_rim_price) / left_rim_price / PATTERN_SETTINGS['cup_and_handle']['rim_tolerance']) * 0.10
        if breakout_idx:
            confidence += 0.05
            
        target_price = rim_level + cup_height
        
        pattern = {
            'type': 'Cup and Handle',
            'left_rim_idx': left_rim_idx,
            'cup_bottom_idx': cup_bottom_idx,
            'right_rim_idx': right_rim_idx,
            'handle_bottom_idx': handle_bottom_idx,
            'breakout_idx': breakout_idx,
            'left_rim_price': float(left_rim_price),
            'cup_bottom_price': float(cup_bottom_price),
            'right_rim_price': float(right_rim_price),
            'handle_bottom_price': float(handle_bottom_price),
            'rim_level': float(rim_level),
            'target_price': float(target_price),
            'cup_height': float(cup_height),
            'cup_depth': cup_depth,
            'handle_retrace': handle_retrace,
            'confidence': min(0.95, confidence),
            'status': 'confirmed' if breakout_idx else 'forming',
            'duration_days': pattern_duration,
            'detection_method': 'traditional'
        }
        
        patterns.append(pattern)
    
    return sorted(patterns, key=lambda x: x['confidence'], reverse=True)

def detect_all_patterns(df, selected_patterns=None, debug=False):
    """Detect all selected patterns"""
    if selected_patterns is None:
        selected_patterns = ["Head and Shoulders", "Double Bottom", "Double Top", "Cup and Handle"]
    
    results = {}
    
    if "Head and Shoulders" in selected_patterns:
        results["Head and Shoulders"] = detect_head_and_shoulders(df, debug=debug)
    
    if "Double Bottom" in selected_patterns:
        results["Double Bottom"] = detect_double_bottom(df, debug=debug)
        
    if "Double Top" in selected_patterns:
        results["Double Top"] = detect_double_top(df, debug=debug)
    
    if "Cup and Handle" in selected_patterns:
        results["Cup and Handle"] = detect_cup_and_handle(df, debug=debug)
    
    return results

# =========================== LIGHTWEIGHT ML DETECTION ===========================

class LightweightMLDetector:
    """Lightweight ML pattern detector using only scikit-learn"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = None
        self.svm_model = None
        self.isolation_forest = None
        self.is_trained = False
        
    def prepare_features(self, df):
        """Extract technical features for ML models"""
        data = df.copy()
        
        # Basic features
        data['returns'] = data['Close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Moving averages
        for period in [5, 10, 20]:
            data[f'ma_{period}'] = data['Close'].rolling(period).mean()
            data[f'ma_ratio_{period}'] = data['Close'] / data[f'ma_{period}']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Price position
        data['high_low_ratio'] = data['High'] / data['Low']
        data['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Volume
        data['volume_ma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma']
        
        # Local extrema
        peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
        troughs = argrelextrema(data['Close'].values, np.less, order=5)[0]
        
        data['local_maxima'] = 0
        data['local_minima'] = 0
        data.iloc[peaks, data.columns.get_loc('local_maxima')] = 1
        data.iloc[troughs, data.columns.get_loc('local_minima')] = 1
        
        return data.fillna(method='ffill').fillna(method='bfill')
    
    def generate_synthetic_patterns(self, n_samples=500):
        """Generate synthetic pattern data for training"""
        patterns = []
        labels = []
        
        for pattern_type in range(4):  # 4 pattern types
            for _ in range(n_samples // 4):
                if pattern_type == 0:  # Head and Shoulders
                    pattern = self.generate_head_shoulders_pattern()
                elif pattern_type == 1:  # Double Bottom
                    pattern = self.generate_double_bottom_pattern()
                elif pattern_type == 2:  # Double Top
                    pattern = self.generate_double_top_pattern()
                else:  # Cup and Handle
                    pattern = self.generate_cup_handle_pattern()
                
                patterns.append(pattern)
                labels.append(pattern_type)
        
        # Return as 2D array (samples, features) instead of 3D
        return np.array(patterns), np.array(labels)
    
    def generate_head_shoulders_pattern(self, length=30):
        """Generate synthetic head and shoulders statistical features"""
        # Generate basic pattern shape
        pattern = np.zeros(length)
        pattern[5:10] = np.linspace(0, 0.6, 5) + np.random.normal(0, 0.05, 5)  # Left shoulder
        pattern[12:17] = np.linspace(0.6, 1.0, 5) + np.random.normal(0, 0.05, 5)  # Head
        pattern[20:25] = np.linspace(1.0, 0.6, 5) + np.random.normal(0, 0.05, 5)  # Right shoulder
        pattern += np.random.normal(0, 0.02, length)
        
        # Extract statistical features (mean, std, min, max)
        features = [
            pattern.mean(), pattern.std(), pattern.min(), pattern.max(),  # Price features
            np.diff(pattern).mean(), np.diff(pattern).std(), np.diff(pattern).min(), np.diff(pattern).max(),  # Returns
            np.random.uniform(0.1, 0.3), np.random.uniform(0.05, 0.15), 0, 1,  # Volatility stats
            np.random.uniform(0.9, 1.1), np.random.uniform(0.05, 0.1), 0.8, 1.2,  # MA ratio
            np.random.uniform(40, 60), np.random.uniform(10, 20), 0, 100,  # RSI
            np.random.uniform(0.8, 1.2), np.random.uniform(0.2, 0.4), 0.5, 2.0,  # Volume
            0.2, 0.4, 0, 1,  # Local maxima (head and shoulders have peaks)
            0.1, 0.3, 0, 1,  # Local minima
            np.random.uniform(0.4, 0.6), np.random.uniform(0.1, 0.2), 0, 1  # Close position
        ]
        return np.array(features[:30])  # Ensure exactly 30 features
    
    def generate_double_bottom_pattern(self, length=30):
        """Generate synthetic double bottom statistical features"""
        pattern = np.ones(length) * 0.5
        pattern[7:12] = np.concatenate([np.linspace(0.5, 0.2, 2), [0.2], np.linspace(0.2, 0.5, 2)]) + np.random.normal(0, 0.03, 5)
        pattern[18:23] = np.concatenate([np.linspace(0.5, 0.2, 2), [0.2], np.linspace(0.2, 0.8, 2)]) + np.random.normal(0, 0.03, 5)
        pattern += np.random.normal(0, 0.02, length)
        
        features = [
            pattern.mean(), pattern.std(), pattern.min(), pattern.max(),
            np.diff(pattern).mean(), np.diff(pattern).std(), np.diff(pattern).min(), np.diff(pattern).max(),
            np.random.uniform(0.15, 0.35), np.random.uniform(0.08, 0.18), 0, 1,
            np.random.uniform(0.85, 1.15), np.random.uniform(0.08, 0.15), 0.7, 1.3,
            np.random.uniform(30, 50), np.random.uniform(8, 18), 0, 100,
            np.random.uniform(1.0, 1.4), np.random.uniform(0.3, 0.5), 0.6, 2.2,
            0.1, 0.3, 0, 1,  # Fewer peaks for double bottom
            0.3, 0.5, 0, 1,  # More troughs for double bottom
            np.random.uniform(0.3, 0.7), np.random.uniform(0.15, 0.25), 0, 1
        ]
        return np.array(features[:30])
    
    def generate_double_top_pattern(self, length=30):
        """Generate synthetic double top statistical features"""
        pattern = np.ones(length) * 0.5
        pattern[7:12] = np.concatenate([np.linspace(0.5, 0.9, 2), [0.9], np.linspace(0.9, 0.5, 2)]) + np.random.normal(0, 0.03, 5)
        pattern[18:23] = np.concatenate([np.linspace(0.5, 0.9, 2), [0.9], np.linspace(0.9, 0.2, 2)]) + np.random.normal(0, 0.03, 5)
        pattern += np.random.normal(0, 0.02, length)
        
        features = [
            pattern.mean(), pattern.std(), pattern.min(), pattern.max(),
            np.diff(pattern).mean(), np.diff(pattern).std(), np.diff(pattern).min(), np.diff(pattern).max(),
            np.random.uniform(0.2, 0.4), np.random.uniform(0.1, 0.2), 0, 1,
            np.random.uniform(0.8, 1.2), np.random.uniform(0.1, 0.2), 0.6, 1.4,
            np.random.uniform(50, 70), np.random.uniform(12, 22), 0, 100,
            np.random.uniform(0.7, 1.1), np.random.uniform(0.2, 0.4), 0.4, 1.8,
            0.3, 0.5, 0, 1,  # More peaks for double top
            0.1, 0.3, 0, 1,  # Fewer troughs
            np.random.uniform(0.4, 0.8), np.random.uniform(0.1, 0.3), 0, 1
        ]
        return np.array(features[:30])
    
    def generate_cup_handle_pattern(self, length=30):
        """Generate synthetic cup and handle statistical features"""
        pattern = np.ones(length) * 0.8
        pattern[5:10] = np.linspace(0.8, 0.3, 5)
        pattern[10:15] = np.linspace(0.3, 0.8, 5)
        pattern[22:27] = np.linspace(0.8, 0.6, 5) + np.random.normal(0, 0.02, 5)
        pattern += np.random.normal(0, 0.02, length)
        
        features = [
            pattern.mean(), pattern.std(), pattern.min(), pattern.max(),
            np.diff(pattern).mean(), np.diff(pattern).std(), np.diff(pattern).min(), np.diff(pattern).max(),
            np.random.uniform(0.1, 0.25), np.random.uniform(0.05, 0.12), 0, 1,
            np.random.uniform(0.95, 1.05), np.random.uniform(0.03, 0.08), 0.9, 1.1,
            np.random.uniform(45, 65), np.random.uniform(8, 15), 0, 100,
            np.random.uniform(1.1, 1.5), np.random.uniform(0.25, 0.45), 0.7, 2.5,
            0.15, 0.35, 0, 1,  # Moderate peaks
            0.2, 0.4, 0, 1,   # Cup bottom creates trough
            np.random.uniform(0.5, 0.9), np.random.uniform(0.1, 0.2), 0, 1
        ]
        return np.array(features[:30])
    
    def train_models(self):
        """Train lightweight ML models with synthetic data"""
        print("Training lightweight ML models...")
        
        # Generate synthetic training data (already 2D with 30 features)
        X_synthetic, y_synthetic = self.generate_synthetic_patterns(1000)
        
        print(f"Training models with {X_synthetic.shape[1]} features...")
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.rf_model.fit(X_synthetic, y_synthetic)
        
        # Train SVM
        X_scaled = self.scaler.fit_transform(X_synthetic)
        self.svm_model = SVC(probability=True, random_state=42)
        self.svm_model.fit(X_scaled, y_synthetic)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(X_synthetic)
        
        self.is_trained = True
        print("Lightweight ML models trained successfully!")
    

    
    def detect_patterns_ml(self, df, confidence_threshold=0.6):
        """Detect patterns using lightweight ML models"""
        if not self.is_trained:
            self.train_models()
        
        # Prepare features
        data = self.prepare_features(df)
        
        # Use simple windowed features instead of long sequences
        window_size = 20
        feature_cols = ['returns', 'volatility', 'ma_ratio_20', 'rsi', 'volume_ratio', 
                       'local_maxima', 'local_minima', 'close_position']
        
        # Ensure we have enough data
        if len(data) < window_size + 10:
            return {}
        
        # Create simple statistical features from windows
        sequences = []
        for i in range(window_size, len(data)):
            window_features = []
            
            for col in feature_cols:
                if col in data.columns:
                    window_data = data[col].iloc[i-window_size:i]
                    # Extract statistical features instead of raw sequence
                    window_features.extend([
                        window_data.mean(),
                        window_data.std(),
                        window_data.min(),
                        window_data.max()
                    ])
            
            # This gives us 8 features * 4 stats = 32 features (close to our synthetic 30)
            if len(window_features) == len(feature_cols) * 4:
                sequences.append(window_features)
        
        if len(sequences) == 0:
            return {}
        
        X = np.array(sequences)
        
        # Ensure we have the right number of features by padding or truncating
        expected_features = 30  # Match our synthetic data
        if X.shape[1] != expected_features:
            if X.shape[1] > expected_features:
                X = X[:, :expected_features]  # Truncate
            else:
                # Pad with zeros
                padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                X = np.hstack([X, padding])
        
        # Get predictions
        rf_pred = self.rf_model.predict_proba(X)
        
        X_scaled = self.scaler.transform(X)
        svm_pred = self.svm_model.predict_proba(X_scaled)
        
        # Ensemble predictions
        ensemble_pred = (rf_pred * 0.6 + svm_pred * 0.4)
        
        # Detect anomalies
        anomalies = self.isolation_forest.predict(X)
        
        # Process predictions
        pattern_names = ['Head and Shoulders', 'Double Bottom', 'Double Top', 'Cup and Handle']
        results = {name: [] for name in pattern_names}
        
        for i, pred in enumerate(ensemble_pred):
            max_confidence = np.max(pred)
            predicted_class = np.argmax(pred)
            
            if max_confidence >= confidence_threshold:
                pattern_type = pattern_names[predicted_class]
                position_idx = i + sequence_length
                
                if position_idx < len(df):
                    pattern_info = {
                        'type': pattern_type,
                        'position_idx': position_idx,
                        'confidence': float(max_confidence),
                        'rf_confidence': float(rf_pred[i][predicted_class]),
                        'svm_confidence': float(svm_pred[i][predicted_class]),
                        'is_anomaly': anomalies[i] == -1,
                        'status': 'ml_detected',
                        'detection_method': 'ml',
                        'price': float(df['Close'].iloc[position_idx]),
                        'target_price': float(df['Close'].iloc[position_idx] * (1.1 if predicted_class in [1, 3] else 0.9)),
                        'date': df.index[position_idx] if hasattr(df.index, 'date') else position_idx
                    }
                    
                    results[pattern_type].append(pattern_info)
        
        # Sort by confidence
        for pattern_type in results:
            results[pattern_type] = sorted(results[pattern_type], 
                                         key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def hybrid_detect_patterns(self, df, use_traditional=True, use_ml=True, confidence_threshold=0.6):
        """Hybrid approach combining traditional and ML detection"""
        all_results = {}
        
        if use_traditional:
            traditional_results = detect_all_patterns(df)
            
            for pattern_type, patterns in traditional_results.items():
                if pattern_type not in all_results:
                    all_results[pattern_type] = []
                
                for pattern in patterns:
                    pattern['detection_method'] = 'traditional'
                    all_results[pattern_type].append(pattern)
        
        if use_ml:
            ml_results = self.detect_patterns_ml(df, confidence_threshold)
            
            for pattern_type, patterns in ml_results.items():
                if pattern_type not in all_results:
                    all_results[pattern_type] = []
                
                for pattern in patterns:
                    all_results[pattern_type].append(pattern)
        
        # Sort by confidence and filter
        for pattern_type in all_results:
            all_results[pattern_type] = sorted(all_results[pattern_type], 
                                             key=lambda x: x['confidence'], reverse=True)
            all_results[pattern_type] = [p for p in all_results[pattern_type] 
                                       if p['confidence'] >= confidence_threshold]
        
        return all_results

# =========================== VISUALIZATION ===========================

def plot_pattern(df, patterns, pattern_type, stock_name="Stock", chart_type="Candlestick"):
    """Plot stock data with pattern annotations"""
    if not patterns:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{stock_name} - {pattern_type}', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Use Date column for x-axis if available
    if 'Date' in df.columns:
        x_axis = df['Date']
    else:
        x_axis = df.index
    
    # Add price chart based on selected type
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=x_axis,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
    elif chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    elif chart_type == "OHLC":
        fig.add_trace(
            go.Ohlc(
                x=x_axis,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
    elif chart_type == "Area":
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                fill='tonexty',
                line=dict(color='rgba(0,100,80,0.8)'),
                fillcolor='rgba(0,100,80,0.2)'
            ),
            row=1, col=1
        )
    
    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(
            x=x_axis, 
            y=df['Volume'], 
            name='Volume', 
            opacity=0.7,
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add pattern annotations
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, pattern in enumerate(patterns[:5]):  # Show max 5 patterns
        color = colors[i % len(colors)]
        
        if pattern_type == "Head and Shoulders":
            # Mark shoulders and head
            fig.add_trace(go.Scatter(
                x=[x_axis.iloc[pattern['left_shoulder_idx']], 
                   x_axis.iloc[pattern['head_idx']], 
                   x_axis.iloc[pattern['right_shoulder_idx']]],
                y=[pattern['left_shoulder_price'], 
                   pattern['head_price'], 
                   pattern['right_shoulder_price']],
                mode='markers+lines',
                name=f'H&S {i+1} (Conf: {pattern["confidence"]:.1%})',
                line=dict(color=color, width=2),
                marker=dict(size=10)
            ), row=1, col=1)
            
            # Neckline
            if 'left_trough_idx' in pattern and 'right_trough_idx' in pattern:
                fig.add_trace(go.Scatter(
                    x=[x_axis.iloc[pattern['left_trough_idx']], 
                       x_axis.iloc[pattern['right_trough_idx']]],
                    y=[df['Close'].iloc[pattern['left_trough_idx']], 
                       df['Close'].iloc[pattern['right_trough_idx']]],
                    mode='lines',
                    name=f'Neckline {i+1}',
                    line=dict(color=color, dash='dash', width=2)
                ), row=1, col=1)
        
        elif pattern_type == "Double Bottom":
            # Mark the two bottoms
            fig.add_trace(go.Scatter(
                x=[x_axis.iloc[pattern['first_trough_idx']], 
                   x_axis.iloc[pattern['second_trough_idx']]],
                y=[pattern['first_trough_price'], 
                   pattern['second_trough_price']],
                mode='markers+lines',
                name=f'DB {i+1} (Conf: {pattern["confidence"]:.1%})',
                line=dict(color=color, width=2),
                marker=dict(size=10)
            ), row=1, col=1)
        
        elif pattern_type == "Double Top":
            # Mark the two tops
            fig.add_trace(go.Scatter(
                x=[x_axis.iloc[pattern['first_peak_idx']], 
                   x_axis.iloc[pattern['second_peak_idx']]],
                y=[pattern['first_peak_price'], 
                   pattern['second_peak_price']],
                mode='markers+lines',
                name=f'DT {i+1} (Conf: {pattern["confidence"]:.1%})',
                line=dict(color=color, width=2),
                marker=dict(size=10)
            ), row=1, col=1)
        
        elif pattern_type == "Cup and Handle":
            # Mark cup and handle points
            if all(key in pattern for key in ['left_rim_idx', 'cup_bottom_idx', 'right_rim_idx', 'handle_bottom_idx']):
                fig.add_trace(go.Scatter(
                    x=[x_axis.iloc[pattern['left_rim_idx']], 
                       x_axis.iloc[pattern['cup_bottom_idx']], 
                       x_axis.iloc[pattern['right_rim_idx']], 
                       x_axis.iloc[pattern['handle_bottom_idx']]],
                    y=[pattern['left_rim_price'], 
                       pattern['cup_bottom_price'], 
                       pattern['right_rim_price'], 
                       pattern['handle_bottom_price']],
                    mode='markers+lines',
                    name=f'C&H {i+1} (Conf: {pattern["confidence"]:.1%})',
                    line=dict(color=color, width=2),
                    marker=dict(size=10)
                ), row=1, col=1)
        
        # Add target price line if available
        if 'target_price' in pattern:
            fig.add_hline(
                y=pattern['target_price'],
                line_dash="dot",
                line_color=color,
                annotation_text=f"Target: ${pattern['target_price']:.2f}",
                row=1, col=1
            )
    
    fig.update_layout(
        title=f"{stock_name} - {pattern_type} Pattern Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True
    )
    
    return fig

# =========================== STREAMLIT UI ===========================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = []
    if 'scan_completed' not in st.session_state:
        st.session_state.scan_completed = False
    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None
    if 'selected_pattern' not in st.session_state:
        st.session_state.selected_pattern = None
    if 'chart_type' not in st.session_state:
        st.session_state.chart_type = "Candlestick"
    if 'ml_detector' not in st.session_state:
        st.session_state.ml_detector = None

def clear_results():
    """Clear all scan results and reset state"""
    st.session_state.scan_results = []
    st.session_state.scan_completed = False
    st.session_state.selected_stock = None
    st.session_state.selected_pattern = None

def main():
    st.set_page_config(
        page_title="Stock Pattern Scanner",
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("📈 Stock Pattern Scanner")
    st.markdown("Advanced pattern detection with traditional algorithms and lightweight ML")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Date range
        with st.expander("📅 Date Range", expanded=True):
            default_start = datetime.date.today() - datetime.timedelta(days=DEFAULT_SCAN_DAYS)
            start_date = st.date_input("Start Date", value=default_start)
            end_date = st.date_input("End Date", value=datetime.date.today())
        
        # Stock selection
        with st.expander("📊 Stock Selection", expanded=True):
            stock_input_method = st.radio(
                "Input Method:",
                ["Individual Stocks", "Upload List", "Predefined Lists"]
            )
            
            if stock_input_method == "Individual Stocks":
                stock_symbols_input = st.text_area(
                    "Enter stock symbols (one per line):",
                    value="AAPL\nMSFT\nGOOGL\nTSLA\nAMZN",
                    height=100
                )
                stock_symbols = [s.strip().upper() for s in stock_symbols_input.split('\n') if s.strip()]
            
            elif stock_input_method == "Upload List":
                uploaded_file = st.file_uploader("Upload CSV/TXT file", type=['csv', 'txt'])
                if uploaded_file:
                    content = uploaded_file.read().decode('utf-8')
                    stock_symbols = [s.strip().upper() for s in content.split('\n') if s.strip()]
                else:
                    stock_symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            else:  # Predefined Lists
                list_choice = st.selectbox("Choose List:", ["Tech Stocks", "S&P 500 Sample"])
                if list_choice == "Tech Stocks":
                    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
                else:
                    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'V', 'PG', 'UNH']
        
        # Pattern selection
        with st.expander("🔍 Pattern Selection", expanded=True):
            detect_head_shoulders = st.checkbox("📉 Head & Shoulders", value=True)
            detect_double_bottom = st.checkbox("📈 Double Bottom", value=True)
            detect_double_top = st.checkbox("📉 Double Top", value=True)
            detect_cup_handle = st.checkbox("☕ Cup & Handle", value=True)
            
            selected_patterns = []
            if detect_head_shoulders:
                selected_patterns.append("Head and Shoulders")
            if detect_double_bottom:
                selected_patterns.append("Double Bottom")
            if detect_double_top:
                selected_patterns.append("Double Top")
            if detect_cup_handle:
                selected_patterns.append("Cup and Handle")
        
        # ML settings
        with st.expander("🤖 ML Settings", expanded=False):
            use_traditional = st.checkbox("📊 Traditional Detection", value=True)
            use_ml = st.checkbox("🧠 ML Detection", value=True)
            ml_confidence_threshold = st.slider(
                "ML Confidence Threshold",
                min_value=0.3,
                max_value=0.9,
                value=0.6,
                step=0.05
            )
            show_ml_details = st.checkbox("Show ML Details", value=False)
    
    # Main scanning interface
    if not selected_patterns:
        st.warning("Please select at least one pattern to detect.")
        return
    
    if not stock_symbols:
        st.warning("Please enter at least one stock symbol.")
        return
    
    # Scanning buttons
    col_scan1, col_scan2 = st.columns([3, 1])
    
    with col_scan1:
        scan_button = st.button("🚀 Start Scanning", type="primary", key="scan_btn")
    
    with col_scan2:
        if st.button("🗑️ Clear Results", key="clear_btn"):
            clear_results()
            st.rerun()
    
    # Always show results section if we have data
    if st.session_state.scan_completed and st.session_state.scan_results:
        display_results_section()
    
    # Handle new scan
    if scan_button:
        # Initialize ML detector if needed
        if use_ml:
            if 'ml_detector' not in st.session_state or st.session_state.ml_detector is None:
                with st.spinner("Initializing ML models..."):
                    st.session_state.ml_detector = LightweightMLDetector()
        else:
            # Set to None if ML is disabled
            st.session_state.ml_detector = None
        
        # Scanning process
        pattern_names = ", ".join(selected_patterns)
        st.info(f"🔍 Scanning {len(stock_symbols)} stocks for: **{pattern_names}**")
        
        progress_bar = st.progress(0)
        results = []
        failed_symbols = []
        
        for i, symbol in enumerate(stock_symbols):
            progress_bar.progress((i + 1) / len(stock_symbols))
            
            try:
                df = fetch_stock_data(symbol, start_date, end_date)
                if df is None or df.empty:
                    failed_symbols.append(symbol)
                    continue
                
                # Pattern detection
                if use_ml and st.session_state.ml_detector is not None:
                    patterns = st.session_state.ml_detector.hybrid_detect_patterns(
                        df,
                        use_traditional=use_traditional,
                        use_ml=use_ml,
                        confidence_threshold=ml_confidence_threshold
                    )
                else:
                    # Use traditional detection only
                    patterns = detect_all_patterns(df, selected_patterns)
                
                # Get current price
                try:
                    current_price = df['Close'].iloc[-1]
                    volume = df['Volume'].iloc[-1]
                except:
                    current_price = None
                    volume = None
                
                # Store results
                stock_result = {
                    'Symbol': symbol,
                    'Current_Price': current_price,
                    'Volume': volume,
                    'Patterns': patterns,
                    'Data': df
                }
                results.append(stock_result)
                
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
                failed_symbols.append(symbol)
        
        progress_bar.empty()
        
        # Store results and mark scan as completed
        if results:
            st.session_state.scan_results = results
            st.session_state.scan_completed = True
            st.success(f"✅ Scan completed! Found patterns in {len(results)} stocks.")
            st.rerun()  # Refresh to show results
        else:
            st.error("No data could be retrieved for the selected stocks.")
        
        if failed_symbols:
            st.warning(f"Failed to process: {', '.join(failed_symbols)}")

def display_results_section():
    """Display the results section with proper state management"""
    results = st.session_state.scan_results
    
    if not results:
        return
    
    # Summary statistics
    total_patterns = sum(len(patterns) for stock in results for patterns in stock['Patterns'].values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stocks Scanned", len(results))
    with col2:
        st.metric("Total Patterns", total_patterns)
    with col3:
        st.metric("Status", "✅ Complete")
    
    # Results table
    st.subheader("📊 Scan Results")
    
    # Create summary dataframe
    summary_data = []
    for stock in results:
        row = {'Symbol': stock['Symbol']}
        if stock['Current_Price']:
            row['Price'] = f"${stock['Current_Price']:.2f}"
        else:
            row['Price'] = "N/A"
        
        # Count patterns for each type
        pattern_counts = {}
        for pattern_name, patterns in stock['Patterns'].items():
            pattern_counts[pattern_name] = len(patterns)
            row[PATTERN_DISPLAY[pattern_name]['icon'] + ' ' + pattern_name] = len(patterns)
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Pattern Analysis Section
    st.subheader("🔍 Pattern Analysis")
    
    # Get stocks with patterns
    stocks_with_patterns = [stock for stock in results if any(len(patterns) > 0 for patterns in stock['Patterns'].values())]
    
    if not stocks_with_patterns:
        st.info("No patterns found in the scanned stocks.")
        return
    
    # Stock selection
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        stock_options = [stock['Symbol'] for stock in stocks_with_patterns]
        
        # Use callback to handle stock selection
        def on_stock_change():
            st.session_state.selected_pattern = None  # Reset pattern when stock changes
        
        selected_stock = st.selectbox(
            "Select Stock:",
            options=stock_options,
            key="stock_select",
            on_change=on_stock_change
        )
        
        st.session_state.selected_stock = selected_stock
    
    # Find selected stock data
    selected_data = next(stock for stock in stocks_with_patterns if stock['Symbol'] == selected_stock)
    
    # Pattern selection
    with col2:
        pattern_options = []
        for pattern_name, patterns in selected_data["Patterns"].items():
            if patterns:
                pattern_options.append(f"{pattern_name} ({len(patterns)})")
        
        if pattern_options:
            selected_pattern_display = st.selectbox(
                "Select Pattern:",
                options=pattern_options,
                key="pattern_select"
            )
            
            # Extract pattern name
            selected_pattern = selected_pattern_display.split(' (')[0]
            st.session_state.selected_pattern = selected_pattern
        else:
            st.info("No patterns found for this stock.")
            return
    
    # Chart type selection
    with col3:
        chart_type = st.selectbox(
            "Chart Type:",
            options=["Candlestick", "Line", "OHLC", "Area"],
            key="chart_type_select"
        )
        st.session_state.chart_type = chart_type
    
    # Display pattern analysis
    if selected_pattern:
        pattern_points = selected_data["Patterns"][selected_pattern]
        
        # Pattern summary
        st.subheader(f"📊 {selected_stock} - {selected_pattern}")
        
        if pattern_points:
            # Summary metrics
            pattern_info_cols = st.columns(4)
            
            with pattern_info_cols[0]:
                st.metric("Patterns Found", len(pattern_points))
            
            with pattern_info_cols[1]:
                avg_confidence = sum(p['confidence'] for p in pattern_points) / len(pattern_points)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with pattern_info_cols[2]:
                if 'duration_days' in pattern_points[0]:
                    avg_duration = sum(p.get('duration_days', 0) for p in pattern_points) / len(pattern_points)
                    st.metric("Avg Duration", f"{avg_duration:.0f} days")
                else:
                    st.metric("Detection Method", pattern_points[0].get('detection_method', 'N/A').title())
            
            with pattern_info_cols[3]:
                confirmed_count = sum(1 for p in pattern_points if p.get('status') == 'confirmed')
                st.metric("Confirmed", f"{confirmed_count}/{len(pattern_points)}")
            
            # Display chart
            fig = plot_pattern(
                selected_data["Data"],
                pattern_points,
                selected_pattern,
                stock_name=selected_data["Symbol"],
                chart_type=chart_type
            )
            
            st.plotly_chart(fig, use_container_width=True, height=600)
            
            # Pattern details
            display_pattern_details(selected_data, selected_pattern, pattern_points)

def display_pattern_details(selected_data, selected_pattern, pattern_points):
    """Display detailed pattern information"""
    st.subheader("🔍 Pattern Details")
    
    for i, pattern in enumerate(pattern_points[:3]):  # Show top 3 patterns
        with st.expander(f"Pattern {i+1} - Confidence: {pattern['confidence']:.1%}", expanded=(i==0)):
            detail_cols = st.columns(3)
            
            with detail_cols[0]:
                st.write("**📅 Key Dates:**")
                display_pattern_dates(selected_data, selected_pattern, pattern)
            
            with detail_cols[1]:
                st.write("**💰 Price Levels:**")
                display_pattern_prices(selected_data, pattern)
            
            with detail_cols[2]:
                st.write("**📊 Pattern Stats:**")
                display_pattern_stats(pattern)

def display_pattern_dates(selected_data, selected_pattern, pattern):
    """Display pattern-specific dates"""
    try:
        if selected_pattern == "Head and Shoulders":
            if 'left_shoulder_idx' in pattern:
                ls_date = selected_data["Data"]['Date'].iloc[pattern['left_shoulder_idx']]
                st.write(f"• Left Shoulder: {ls_date.strftime('%Y-%m-%d')}")
            if 'head_idx' in pattern:
                head_date = selected_data["Data"]['Date'].iloc[pattern['head_idx']]
                st.write(f"• Head: {head_date.strftime('%Y-%m-%d')}")
            if 'right_shoulder_idx' in pattern:
                rs_date = selected_data["Data"]['Date'].iloc[pattern['right_shoulder_idx']]
                st.write(f"• Right Shoulder: {rs_date.strftime('%Y-%m-%d')}")
            if 'breakout_idx' in pattern and pattern['breakout_idx']:
                bo_date = selected_data["Data"]['Date'].iloc[pattern['breakout_idx']]
                st.write(f"• Breakout: {bo_date.strftime('%Y-%m-%d')}")
        
        elif selected_pattern == "Double Bottom":
            if 'first_trough_idx' in pattern:
                t1_date = selected_data["Data"]['Date'].iloc[pattern['first_trough_idx']]
                st.write(f"• First Trough: {t1_date.strftime('%Y-%m-%d')}")
            if 'second_trough_idx' in pattern:
                t2_date = selected_data["Data"]['Date'].iloc[pattern['second_trough_idx']]
                st.write(f"• Second Trough: {t2_date.strftime('%Y-%m-%d')}")
            if 'neckline_idx' in pattern:
                neck_date = selected_data["Data"]['Date'].iloc[pattern['neckline_idx']]
                st.write(f"• Neckline: {neck_date.strftime('%Y-%m-%d')}")
            if 'breakout_idx' in pattern and pattern['breakout_idx']:
                bo_date = selected_data["Data"]['Date'].iloc[pattern['breakout_idx']]
                st.write(f"• Breakout: {bo_date.strftime('%Y-%m-%d')}")
        
        elif selected_pattern == "Double Top":
            if 'first_peak_idx' in pattern:
                p1_date = selected_data["Data"]['Date'].iloc[pattern['first_peak_idx']]
                st.write(f"• First Peak: {p1_date.strftime('%Y-%m-%d')}")
            if 'second_peak_idx' in pattern:
                p2_date = selected_data["Data"]['Date'].iloc[pattern['second_peak_idx']]
                st.write(f"• Second Peak: {p2_date.strftime('%Y-%m-%d')}")
            if 'neckline_idx' in pattern:
                neck_date = selected_data["Data"]['Date'].iloc[pattern['neckline_idx']]
                st.write(f"• Neckline: {neck_date.strftime('%Y-%m-%d')}")
            if 'breakdown_idx' in pattern and pattern['breakdown_idx']:
                bd_date = selected_data["Data"]['Date'].iloc[pattern['breakdown_idx']]
                st.write(f"• Breakdown: {bd_date.strftime('%Y-%m-%d')}")
        
        elif selected_pattern == "Cup and Handle":
            if 'left_rim_idx' in pattern:
                lr_date = selected_data["Data"]['Date'].iloc[pattern['left_rim_idx']]
                st.write(f"• Left Rim: {lr_date.strftime('%Y-%m-%d')}")
            if 'cup_bottom_idx' in pattern:
                cb_date = selected_data["Data"]['Date'].iloc[pattern['cup_bottom_idx']]
                st.write(f"• Cup Bottom: {cb_date.strftime('%Y-%m-%d')}")
            if 'right_rim_idx' in pattern:
                rr_date = selected_data["Data"]['Date'].iloc[pattern['right_rim_idx']]
                st.write(f"• Right Rim: {rr_date.strftime('%Y-%m-%d')}")
            if 'handle_bottom_idx' in pattern:
                hb_date = selected_data["Data"]['Date'].iloc[pattern['handle_bottom_idx']]
                st.write(f"• Handle Bottom: {hb_date.strftime('%Y-%m-%d')}")
            if 'breakout_idx' in pattern and pattern['breakout_idx']:
                bo_date = selected_data["Data"]['Date'].iloc[pattern['breakout_idx']]
                st.write(f"• Breakout: {bo_date.strftime('%Y-%m-%d')}")
        
        # ML detection dates
        elif pattern.get('detection_method') == 'ml' and 'position_idx' in pattern:
            pos_date = selected_data["Data"]['Date'].iloc[pattern['position_idx']]
            st.write(f"• Detected: {pos_date.strftime('%Y-%m-%d')}")
    
    except Exception as e:
        st.write("• Date information unavailable")

def display_pattern_prices(selected_data, pattern):
    """Display pattern price information"""
    if 'target_price' in pattern:
        st.write(f"• Target: ${pattern['target_price']:.2f}")
    if 'neckline_price' in pattern:
        st.write(f"• Neckline: ${pattern['neckline_price']:.2f}")
    elif 'neckline_level' in pattern:
        st.write(f"• Neckline: ${pattern['neckline_level']:.2f}")
    if 'pattern_height' in pattern:
        st.write(f"• Height: ${pattern['pattern_height']:.2f}")
    
    current_price = selected_data["Data"]['Close'].iloc[-1]
    st.write(f"• Current: ${current_price:.2f}")

def display_pattern_stats(pattern):
    """Display pattern statistics"""
    st.write(f"• Confidence: {pattern['confidence']:.1%}")
    st.write(f"• Status: {pattern.get('status', 'N/A').title()}")
    st.write(f"• Method: {pattern.get('detection_method', 'N/A').title()}")
    if 'duration_days' in pattern:
        st.write(f"• Duration: {pattern['duration_days']} days")
if __name__ == "__main__":
    main()