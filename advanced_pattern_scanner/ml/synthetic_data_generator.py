"""
Synthetic Data Generator for Pattern Training.

This module generates synthetic stock market data with known patterns
based on the reference algorithm specifications. This data is used to
train the CNN-LSTM model for pattern validation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
import logging

from ..core.models import Pattern, MarketData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates synthetic market data with embedded patterns.
    
    This class creates realistic stock price movements with known patterns
    that match the specifications from reference documents (HnS.md, 
    doubleBottom.md, CupHandle.md).
    """
    
    def __init__(self, base_price: float = 100.0, volatility: float = 0.02):
        """
        Initialize the synthetic data generator.
        
        Args:
            base_price: Base stock price for generation
            volatility: Daily volatility (standard deviation of returns)
        """
        self.base_price = base_price
        self.volatility = volatility
        
        # Pattern templates based on reference specifications
        self.pattern_templates = {
            "Head and Shoulders": self._generate_head_shoulders_template,
            "Double Bottom": self._generate_double_bottom_template,
            "Double Top": self._generate_double_top_template,
            "Cup and Handle": self._generate_cup_handle_template,
            "Ascending Triangle": self._generate_ascending_triangle_template,
            "No Pattern": self._generate_random_walk_template
        }
        
        logger.info("SyntheticDataGenerator initialized")
    
    def generate_dataset(self, num_samples: int = 1000, 
                        sequence_length: int = 120) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate a complete dataset for training.
        
        Args:
            num_samples: Number of samples to generate
            sequence_length: Length of each price sequence
            
        Returns:
            Tuple of (features, labels, pattern_names)
        """
        features = []
        labels = []
        pattern_names = list(self.pattern_templates.keys())
        
        samples_per_pattern = num_samples // len(pattern_names)
        
        for pattern_idx, pattern_name in enumerate(pattern_names):
            logger.info(f"Generating {samples_per_pattern} samples for {pattern_name}")
            
            for _ in range(samples_per_pattern):
                # Generate base price sequence
                prices, volumes = self._generate_base_sequence(sequence_length)
                
                # Embed pattern
                if pattern_name != "No Pattern":
                    prices, volumes = self._embed_pattern(prices, volumes, pattern_name)
                
                # Calculate features
                sample_features = self._calculate_features(prices, volumes)
                
                features.append(sample_features)
                labels.append(pattern_idx)
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        
        logger.info(f"Generated dataset: {features.shape[0]} samples, {features.shape[1]} features")
        
        return features, labels, pattern_names
    
    def _generate_base_sequence(self, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate base OHLCV sequence using geometric Brownian motion.
        
        Args:
            length: Sequence length
            
        Returns:
            Tuple of (prices, volumes) as OHLCV and volume arrays
        """
        # Generate returns using geometric Brownian motion
        dt = 1.0  # Daily time step
        drift = 0.0001  # Small positive drift
        
        returns = np.random.normal(drift * dt, self.volatility * np.sqrt(dt), length)
        
        # Generate price levels
        price_levels = self.base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from price levels
        prices = np.zeros((length, 5))  # OHLCV format
        
        for i in range(length):
            # Current close price
            close = price_levels[i]
            
            # Generate intraday volatility
            intraday_vol = self.volatility * 0.5
            high_factor = 1 + abs(np.random.normal(0, intraday_vol))
            low_factor = 1 - abs(np.random.normal(0, intraday_vol))
            
            # Calculate OHLC
            if i == 0:
                open_price = close
            else:
                open_price = prices[i-1, 3]  # Previous close
            
            high = max(open_price, close) * high_factor
            low = min(open_price, close) * low_factor
            
            prices[i] = [open_price, high, low, close, 0]  # Volume set separately
        
        # Generate volumes (log-normal distribution)
        base_volume = 1000000
        volume_volatility = 0.3
        volumes = np.random.lognormal(
            np.log(base_volume) - volume_volatility**2/2, 
            volume_volatility, 
            length
        )
        
        prices[:, 4] = volumes
        
        return prices, volumes
    
    def _embed_pattern(self, prices: np.ndarray, volumes: np.ndarray, 
                      pattern_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed a specific pattern into the price sequence.
        
        Args:
            prices: Base OHLCV array
            volumes: Volume array
            pattern_name: Name of pattern to embed
            
        Returns:
            Modified (prices, volumes) with embedded pattern
        """
        if pattern_name in self.pattern_templates:
            template_func = self.pattern_templates[pattern_name]
            return template_func(prices, volumes)
        else:
            return prices, volumes
    
    def _generate_head_shoulders_template(self, prices: np.ndarray, 
                                        volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Head and Shoulders pattern based on HnS.md specifications."""
        length = len(prices)
        if length < 60:  # Need sufficient length for pattern
            return prices, volumes
        
        # Pattern placement (middle 60% of sequence)
        start_idx = int(length * 0.2)
        end_idx = int(length * 0.8)
        pattern_length = end_idx - start_idx
        
        # Define pattern structure (based on HnS.md)
        left_shoulder_idx = start_idx + int(pattern_length * 0.15)
        head_idx = start_idx + int(pattern_length * 0.5)
        right_shoulder_idx = start_idx + int(pattern_length * 0.85)
        
        # Get base price level
        base_price = prices[start_idx, 3]  # Close price
        
        # Pattern parameters
        shoulder_height = base_price * 1.08  # 8% above base
        head_height = base_price * 1.15     # 15% above base
        neckline_level = base_price * 1.02  # 2% above base
        
        # Create pattern points
        pattern_points = [
            (start_idx, base_price),
            (left_shoulder_idx, shoulder_height),
            (start_idx + int(pattern_length * 0.32), neckline_level),
            (head_idx, head_height),
            (start_idx + int(pattern_length * 0.68), neckline_level),
            (right_shoulder_idx, shoulder_height),
            (end_idx, base_price * 0.95)  # Breakdown
        ]
        
        # Interpolate pattern
        modified_prices = prices.copy()
        for i in range(len(pattern_points) - 1):
            start_point = pattern_points[i]
            end_point = pattern_points[i + 1]
            
            # Linear interpolation between points
            x_range = np.arange(start_point[0], end_point[0] + 1)
            if len(x_range) > 1:
                y_interp = np.linspace(start_point[1], end_point[1], len(x_range))
                
                for j, idx in enumerate(x_range):
                    if idx < length:
                        # Modify OHLC maintaining relationships
                        close_price = y_interp[j]
                        open_price = modified_prices[idx, 0]
                        
                        # Adjust high and low
                        high = max(open_price, close_price) * (1 + np.random.uniform(0, 0.01))
                        low = min(open_price, close_price) * (1 - np.random.uniform(0, 0.01))
                        
                        modified_prices[idx] = [open_price, high, low, close_price, modified_prices[idx, 4]]
        
        # Increase volume at key points
        volumes[left_shoulder_idx] *= 1.5
        volumes[head_idx] *= 2.0
        volumes[right_shoulder_idx] *= 1.3
        volumes[end_idx-1:end_idx+2] *= 1.8  # Breakdown volume
        
        return modified_prices, volumes
    
    def _generate_double_bottom_template(self, prices: np.ndarray, 
                                       volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Double Bottom pattern based on doubleBottom.md specifications."""
        length = len(prices)
        if length < 40:
            return prices, volumes
        
        # Pattern placement
        start_idx = int(length * 0.2)
        end_idx = int(length * 0.8)
        pattern_length = end_idx - start_idx
        
        # Define pattern structure
        first_bottom_idx = start_idx + int(pattern_length * 0.25)
        peak_idx = start_idx + int(pattern_length * 0.5)
        second_bottom_idx = start_idx + int(pattern_length * 0.75)
        
        # Get base price level
        base_price = prices[start_idx, 3]
        
        # Pattern parameters (2% tolerance as per doubleBottom.md)
        bottom_price = base_price * 0.90  # 10% decline
        peak_price = base_price * 1.05    # 5% recovery
        tolerance = 0.02  # 2% tolerance between bottoms
        
        second_bottom_price = bottom_price * (1 + np.random.uniform(-tolerance, tolerance))
        
        # Create pattern
        pattern_points = [
            (start_idx, base_price),
            (first_bottom_idx, bottom_price),
            (peak_idx, peak_price),
            (second_bottom_idx, second_bottom_price),
            (end_idx, base_price * 1.08)  # Breakout
        ]
        
        # Apply pattern
        modified_prices = prices.copy()
        for i in range(len(pattern_points) - 1):
            start_point = pattern_points[i]
            end_point = pattern_points[i + 1]
            
            x_range = np.arange(start_point[0], end_point[0] + 1)
            if len(x_range) > 1:
                y_interp = np.linspace(start_point[1], end_point[1], len(x_range))
                
                for j, idx in enumerate(x_range):
                    if idx < length:
                        close_price = y_interp[j]
                        open_price = modified_prices[idx, 0]
                        
                        high = max(open_price, close_price) * (1 + np.random.uniform(0, 0.005))
                        low = min(open_price, close_price) * (1 - np.random.uniform(0, 0.005))
                        
                        modified_prices[idx] = [open_price, high, low, close_price, modified_prices[idx, 4]]
        
        # Volume pattern (higher at bottoms, breakout)
        volumes[first_bottom_idx-1:first_bottom_idx+2] *= 1.6
        volumes[second_bottom_idx-1:second_bottom_idx+2] *= 1.4
        volumes[end_idx-2:end_idx+1] *= 2.0  # Breakout volume
        
        return modified_prices, volumes
    
    def _generate_double_top_template(self, prices: np.ndarray, 
                                    volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Double Top pattern (inverse of double bottom)."""
        length = len(prices)
        if length < 40:
            return prices, volumes
        
        start_idx = int(length * 0.2)
        end_idx = int(length * 0.8)
        pattern_length = end_idx - start_idx
        
        first_top_idx = start_idx + int(pattern_length * 0.25)
        trough_idx = start_idx + int(pattern_length * 0.5)
        second_top_idx = start_idx + int(pattern_length * 0.75)
        
        base_price = prices[start_idx, 3]
        
        top_price = base_price * 1.12  # 12% rise
        trough_price = base_price * 1.05  # 5% pullback
        tolerance = 0.02
        
        second_top_price = top_price * (1 + np.random.uniform(-tolerance, tolerance))
        
        pattern_points = [
            (start_idx, base_price),
            (first_top_idx, top_price),
            (trough_idx, trough_price),
            (second_top_idx, second_top_price),
            (end_idx, base_price * 0.95)  # Breakdown
        ]
        
        # Apply pattern (similar to double bottom)
        modified_prices = prices.copy()
        for i in range(len(pattern_points) - 1):
            start_point = pattern_points[i]
            end_point = pattern_points[i + 1]
            
            x_range = np.arange(start_point[0], end_point[0] + 1)
            if len(x_range) > 1:
                y_interp = np.linspace(start_point[1], end_point[1], len(x_range))
                
                for j, idx in enumerate(x_range):
                    if idx < length:
                        close_price = y_interp[j]
                        open_price = modified_prices[idx, 0]
                        
                        high = max(open_price, close_price) * (1 + np.random.uniform(0, 0.005))
                        low = min(open_price, close_price) * (1 - np.random.uniform(0, 0.005))
                        
                        modified_prices[idx] = [open_price, high, low, close_price, modified_prices[idx, 4]]
        
        # Volume pattern
        volumes[first_top_idx-1:first_top_idx+2] *= 1.8
        volumes[second_top_idx-1:second_top_idx+2] *= 1.5
        volumes[end_idx-2:end_idx+1] *= 2.2  # Breakdown volume
        
        return modified_prices, volumes
    
    def _generate_cup_handle_template(self, prices: np.ndarray, 
                                    volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Cup and Handle pattern based on CupHandle.md specifications."""
        length = len(prices)
        if length < 80:  # Cup and handle needs longer formation
            return prices, volumes
        
        start_idx = int(length * 0.1)
        end_idx = int(length * 0.9)
        pattern_length = end_idx - start_idx
        
        # Cup formation (70% of pattern)
        cup_end_idx = start_idx + int(pattern_length * 0.7)
        cup_bottom_idx = start_idx + int(pattern_length * 0.35)
        
        # Handle formation (30% of pattern)
        handle_start_idx = cup_end_idx
        handle_end_idx = end_idx
        
        base_price = prices[start_idx, 3]
        
        # Cup parameters (15-50% depth as per CupHandle.md)
        cup_depth = np.random.uniform(0.15, 0.35)  # 15-35% depth
        cup_bottom_price = base_price * (1 - cup_depth)
        
        # Handle parameters (shallow pullback)
        handle_depth = np.random.uniform(0.05, 0.15)  # 5-15% pullback
        handle_low_price = base_price * (1 - handle_depth)
        
        # Create cup (U-shaped)
        cup_points = []
        cup_indices = np.linspace(start_idx, cup_end_idx, 20)
        
        for i, idx in enumerate(cup_indices):
            # Create U-shape using quadratic function
            t = i / (len(cup_indices) - 1)  # Normalize to 0-1
            u_factor = 4 * t * (1 - t)  # Quadratic U-shape
            price = base_price - (base_price - cup_bottom_price) * u_factor
            cup_points.append((int(idx), price))
        
        # Create handle (slight downward trend then breakout)
        handle_points = [
            (handle_start_idx, base_price),
            (handle_start_idx + int((handle_end_idx - handle_start_idx) * 0.6), handle_low_price),
            (handle_end_idx, base_price * 1.08)  # Breakout
        ]
        
        # Apply cup pattern
        modified_prices = prices.copy()
        all_points = cup_points + handle_points
        
        for i in range(len(all_points) - 1):
            start_point = all_points[i]
            end_point = all_points[i + 1]
            
            x_range = np.arange(start_point[0], end_point[0] + 1)
            if len(x_range) > 1:
                y_interp = np.linspace(start_point[1], end_point[1], len(x_range))
                
                for j, idx in enumerate(x_range):
                    if idx < length:
                        close_price = y_interp[j]
                        open_price = modified_prices[idx, 0]
                        
                        high = max(open_price, close_price) * (1 + np.random.uniform(0, 0.003))
                        low = min(open_price, close_price) * (1 - np.random.uniform(0, 0.003))
                        
                        modified_prices[idx] = [open_price, high, low, close_price, modified_prices[idx, 4]]
        
        # Volume pattern (decreasing in cup, increasing on breakout)
        cup_volume_factor = np.linspace(1.0, 0.6, cup_end_idx - start_idx)
        volumes[start_idx:cup_end_idx] *= cup_volume_factor
        volumes[handle_end_idx-2:handle_end_idx+1] *= 2.5  # Breakout volume
        
        return modified_prices, volumes
    
    def _generate_ascending_triangle_template(self, prices: np.ndarray, 
                                            volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Ascending Triangle pattern."""
        length = len(prices)
        if length < 50:
            return prices, volumes
        
        start_idx = int(length * 0.2)
        end_idx = int(length * 0.8)
        
        base_price = prices[start_idx, 3]
        resistance_level = base_price * 1.10  # 10% resistance
        
        # Create ascending lows and flat highs
        num_touches = 4
        touch_indices = np.linspace(start_idx, end_idx - 10, num_touches)
        
        modified_prices = prices.copy()
        
        for i, idx in enumerate(touch_indices):
            idx = int(idx)
            if i % 2 == 0:  # Low touches (ascending)
                low_price = base_price * (0.95 + 0.03 * i / num_touches)
                modified_prices[idx, 3] = low_price  # Close
                modified_prices[idx, 2] = low_price * 0.998  # Low
            else:  # High touches (resistance)
                modified_prices[idx, 3] = resistance_level * 0.999
                modified_prices[idx, 1] = resistance_level  # High
        
        # Breakout
        breakout_idx = end_idx - 5
        modified_prices[breakout_idx:end_idx, 3] = resistance_level * 1.05
        
        # Volume pattern
        volumes[int(touch_indices[-1]):end_idx] *= 1.8
        
        return modified_prices, volumes
    
    def _generate_random_walk_template(self, prices: np.ndarray, 
                                     volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random walk (no pattern)."""
        # Return original prices with some additional noise
        noise_factor = 0.005
        noise = np.random.normal(0, noise_factor, len(prices))
        
        modified_prices = prices.copy()
        for i in range(len(prices)):
            price_multiplier = 1 + noise[i]
            modified_prices[i, :4] *= price_multiplier  # Apply to OHLC
        
        return modified_prices, volumes
    
    def _calculate_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Calculate features from OHLCV data for ML model.
        
        Args:
            prices: OHLCV array
            volumes: Volume array
            
        Returns:
            Feature array for ML model
        """
        closes = prices[:, 3]
        highs = prices[:, 1]
        lows = prices[:, 2]
        
        # Normalize prices and volumes
        if len(closes) > 0:
            price_norm = closes / np.mean(closes)
            volume_norm = volumes / np.mean(volumes)
        else:
            price_norm = closes
            volume_norm = volumes
        
        # Take last 60 periods for sequence
        sequence_length = min(60, len(price_norm))
        
        # Create feature matrix (sequence_length, num_features)
        features = np.zeros((60, 10))  # Pad to 60 if necessary
        
        if sequence_length > 0:
            start_idx = max(0, len(price_norm) - 60)
            
            # OHLCV features (normalized)
            features[:sequence_length, 0] = price_norm[start_idx:]  # Close
            features[:sequence_length, 1] = highs[start_idx:] / np.mean(closes) if len(closes) > 0 else 0
            features[:sequence_length, 2] = lows[start_idx:] / np.mean(closes) if len(closes) > 0 else 0
            features[:sequence_length, 3] = volume_norm[start_idx:]
            
            # Technical indicators
            if sequence_length > 1:
                returns = np.diff(price_norm[start_idx:])
                features[1:sequence_length, 4] = returns
                
                # Simple moving averages
                if sequence_length >= 5:
                    sma_5 = np.convolve(price_norm[start_idx:], np.ones(5)/5, mode='valid')
                    features[4:4+len(sma_5), 5] = sma_5
                
                if sequence_length >= 10:
                    sma_10 = np.convolve(price_norm[start_idx:], np.ones(10)/10, mode='valid')
                    features[9:9+len(sma_10), 6] = sma_10
                
                # Volatility (rolling standard deviation)
                if sequence_length >= 10:
                    for i in range(9, sequence_length):
                        window_returns = returns[max(0, i-9):i+1]
                        features[i, 7] = np.std(window_returns) if len(window_returns) > 1 else 0
                
                # Price momentum
                if sequence_length >= 5:
                    for i in range(4, sequence_length):
                        features[i, 8] = (price_norm[start_idx + i] - price_norm[start_idx + i - 4]) / price_norm[start_idx + i - 4]
                
                # Volume momentum
                if sequence_length >= 5:
                    for i in range(4, sequence_length):
                        features[i, 9] = (volume_norm[start_idx + i] - volume_norm[start_idx + i - 4]) / (volume_norm[start_idx + i - 4] + 1e-10)
        
        return features.flatten()  # Flatten for model input