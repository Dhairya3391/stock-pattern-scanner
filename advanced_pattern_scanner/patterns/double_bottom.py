"""
Double Bottom Pattern Detection Algorithm.

Implements the exact methodology from doubleBottom.md using scipy.signal.find_peaks
with 2% tolerance validation and the specific examples from Stock ABC/XYZ.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import logging
from scipy.signal import find_peaks

from ..core.models import Pattern, PatternConfig, MarketData

logger = logging.getLogger(__name__)


class DoubleBottomDetector:
    """
    Detects Double Bottom patterns using the reference algorithm.
    
    Implements the exact methodology from doubleBottom.md:
    1. Identify local minima and maxima using scipy.signal.find_peaks
    2. Select pairs of similar minima (within 2% tolerance)
    3. Check for intervening maximum
    4. Verify no significant lower lows
    5. Determine resistance level
    6. Confirm breakout above resistance
    """
    
    def __init__(self, config: PatternConfig):
        """Initialize detector with configuration."""
        self.config = config
        self.similarity_threshold = config.double_pattern_tolerance  # 2% from reference
        self.min_distance = 3   # Minimum 3 days between minima (adjustable for short sequences)
        self.tolerance = 0.01   # 1% tolerance for lower lows
        self.peak_threshold = 0.02  # 2% minimum peak height
    
    def find_local_extrema(self, prices: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Find local minima and maxima using scipy.signal.find_peaks.
        
        Args:
            prices: Array of price values
            
        Returns:
            Tuple of (minima_indices, maxima_indices)
        """
        # Find minima by inverting prices
        minima_indices, _ = find_peaks(-prices, distance=self.min_distance)
        
        # Find maxima directly
        maxima_indices, _ = find_peaks(prices, distance=self.min_distance)
        
        logger.debug(f"Found {len(minima_indices)} minima and {len(maxima_indices)} maxima")
        return minima_indices.tolist(), maxima_indices.tolist()
    
    def check_similarity(self, prices: np.ndarray, i: int, j: int) -> bool:
        """
        Check if two bottoms are similar within tolerance.
        
        Args:
            prices: Price array
            i: First bottom index
            j: Second bottom index
            
        Returns:
            True if bottoms are similar within 2% tolerance
        """
        price_diff = abs(prices[i] - prices[j])
        min_price = min(prices[i], prices[j])
        
        if min_price == 0:
            return False
        
        similarity_ratio = price_diff / min_price
        is_similar = similarity_ratio <= self.similarity_threshold
        
        logger.debug(f"Similarity check: {prices[i]:.2f} vs {prices[j]:.2f}, "
                    f"ratio={similarity_ratio:.4f}, similar={is_similar}")
        
        return is_similar
    
    def find_intervening_maximum(self, maxima: List[int], i: int, j: int) -> Optional[int]:
        """
        Find maximum between two minima indices.
        
        Args:
            maxima: List of maxima indices
            i: First minimum index
            j: Second minimum index
            
        Returns:
            Index of intervening maximum or None if not found
        """
        intervening_maxima = [k for k in maxima if i < k < j]
        
        if not intervening_maxima:
            return None
        
        # Return the highest maximum if multiple exist
        return max(intervening_maxima)
    
    def check_no_lower_lows(self, prices: np.ndarray, i: int, j: int) -> bool:
        """
        Verify no significant lower lows between the two bottoms.
        
        Args:
            prices: Price array
            i: First bottom index
            j: Second bottom index
            
        Returns:
            True if no significant lower lows exist
        """
        if j <= i + 1:
            return True  # No points between i and j
        
        min_between = np.min(prices[i + 1:j])
        min_bottoms = min(prices[i], prices[j])
        threshold = min_bottoms * (1 - self.tolerance)  # 1% tolerance
        
        no_lower_lows = min_between >= threshold
        
        logger.debug(f"Lower lows check: min_between={min_between:.2f}, "
                    f"threshold={threshold:.2f}, passed={no_lower_lows}")
        
        return no_lower_lows
    
    def calculate_resistance_level(self, prices: np.ndarray, i: int, j: int) -> float:
        """
        Calculate resistance level as maximum price between bottoms.
        
        Args:
            prices: Price array
            i: First bottom index
            j: Second bottom index
            
        Returns:
            Resistance level (highest price between bottoms)
        """
        resistance = np.max(prices[i:j + 1])
        logger.debug(f"Resistance level: {resistance:.2f}")
        return resistance
    
    def validate_peak_height(self, prices: np.ndarray, resistance: float, 
                           i: int, j: int) -> bool:
        """
        Validate that peak is sufficiently pronounced.
        
        Args:
            prices: Price array
            resistance: Resistance level
            i: First bottom index
            j: Second bottom index
            
        Returns:
            True if peak height meets threshold
        """
        max_bottom = max(prices[i], prices[j])
        required_height = max_bottom * (1 + self.peak_threshold)
        
        is_valid = resistance > required_height
        
        logger.debug(f"Peak height validation: resistance={resistance:.2f}, "
                    f"required={required_height:.2f}, valid={is_valid}")
        
        return is_valid
    
    def find_breakout(self, prices: np.ndarray, resistance: float, j: int) -> Optional[int]:
        """
        Find breakout above resistance level after second bottom.
        
        Args:
            prices: Price array
            resistance: Resistance level to break
            j: Second bottom index
            
        Returns:
            Index of breakout or None if not found
        """
        for m in range(j + 1, len(prices)):
            if prices[m] > resistance:
                logger.debug(f"Breakout found at index {m}: {prices[m]:.2f} > {resistance:.2f}")
                return m
        
        return None
    
    def detect_pattern(self, market_data: MarketData) -> List[Pattern]:
        """
        Main detection method implementing the complete double bottom algorithm.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            List of detected Double Bottom patterns
        """
        patterns = []
        
        try:
            prices = market_data.close_prices
            volumes = market_data.volumes
            timestamps = market_data.timestamps
            
            if len(prices) < 10:  # Need minimum data for pattern
                logger.warning(f"Insufficient data for double bottom detection: {len(prices)} points")
                return patterns
            
            # Step 1: Find local minima and maxima
            minima, maxima = self.find_local_extrema(prices)
            
            if len(minima) < 2:
                logger.debug("Insufficient minima for double bottom pattern")
                return patterns
            
            # Step 2: Process each pair of minima
            for idx1 in range(len(minima)):
                for idx2 in range(idx1 + 1, len(minima)):
                    i, j = minima[idx1], minima[idx2]
                    
                    # Skip if too close together
                    if j - i < self.min_distance:
                        continue
                    
                    logger.debug(f"Analyzing minima pair: {i}({prices[i]:.2f}) and {j}({prices[j]:.2f})")
                    
                    # Step 2a: Check similarity
                    if not self.check_similarity(prices, i, j):
                        continue
                    
                    # Step 2b: Define resistance level
                    resistance = self.calculate_resistance_level(prices, i, j)
                    
                    # Step 2c: Validate peak height
                    if not self.validate_peak_height(prices, resistance, i, j):
                        continue
                    
                    # Step 3: Check for intervening maximum
                    peak_index = self.find_intervening_maximum(maxima, i, j)
                    if peak_index is None:
                        logger.debug("No intervening maximum found")
                        continue
                    
                    # Step 4: Check for lower lows
                    if not self.check_no_lower_lows(prices, i, j):
                        continue
                    
                    # Step 5: Find breakout
                    breakout_index = self.find_breakout(prices, resistance, j)
                    
                    if breakout_index is not None:
                        # Calculate pattern metrics
                        pattern_height = resistance - min(prices[i], prices[j])
                        target_price = resistance + pattern_height  # Measured move
                        
                        # Volume analysis (optional enhancement)
                        volume_ratio = 1.0
                        if breakout_index < len(volumes):
                            avg_volume = np.mean(volumes[max(0, j-20):j+1])
                            if avg_volume > 0:
                                volume_ratio = volumes[breakout_index] / avg_volume
                        
                        volume_confirmed = volume_ratio > self.config.min_volume_ratio
                        
                        # Calculate confidence based on pattern quality
                        confidence = 0.7
                        if volume_confirmed:
                            confidence += 0.1
                        if pattern_height / prices[i] > 0.1:  # Significant pattern height
                            confidence += 0.1
                        
                        # Create pattern object
                        pattern = Pattern(
                            type="Double Bottom",
                            symbol=market_data.symbol,
                            timeframe=market_data.timeframe,
                            key_points=[
                                (i, prices[i]),           # First bottom
                                (peak_index, prices[peak_index]),  # Peak
                                (j, prices[j]),           # Second bottom
                                (breakout_index, prices[breakout_index])  # Breakout
                            ],
                            confidence=min(confidence, 1.0),
                            traditional_score=0.8,
                            combined_score=min(confidence, 1.0),
                            entry_price=prices[breakout_index],
                            target_price=target_price,
                            stop_loss=min(prices[i], prices[j]) * 0.98,  # 2% below lowest bottom
                            risk_reward_ratio=(target_price - prices[breakout_index]) / 
                                            (prices[breakout_index] - min(prices[i], prices[j]) * 0.98),
                            formation_start=timestamps[i],
                            formation_end=timestamps[j],
                            breakout_date=timestamps[breakout_index],
                            status="confirmed",
                            volume_confirmation=volume_confirmed,
                            avg_volume_ratio=volume_ratio,
                            pattern_height=pattern_height,
                            duration_days=(timestamps[j] - timestamps[i]).days,
                            detection_method="traditional"
                        )
                        
                        patterns.append(pattern)
                        logger.info(f"Detected Double Bottom for {market_data.symbol}: "
                                   f"Bottom1={timestamps[i].date()}({prices[i]:.2f}), "
                                   f"Bottom2={timestamps[j].date()}({prices[j]:.2f}), "
                                   f"Breakout={timestamps[breakout_index].date()}({prices[breakout_index]:.2f}), "
                                   f"Target={target_price:.2f}")
                        
                        # Only return first valid pattern to avoid overlaps
                        break
                
                if patterns:  # Found a pattern, stop searching
                    break
        
        except Exception as e:
            logger.error(f"Error in Double Bottom detection for {market_data.symbol}: {e}")
        
        return patterns
    
    def test_with_reference_examples(self):
        """
        Test the algorithm with the exact examples from doubleBottom.md.
        
        This method validates the implementation against Stock ABC and XYZ examples.
        """
        logger.info("Testing Double Bottom detector with reference examples")
        
        # Example 1: Stock ABC (should detect pattern)
        abc_prices = np.array([110, 105, 100, 105, 110, 115, 120, 115, 110, 105, 102, 105, 110, 115, 125])
        abc_timestamps = [datetime(2024, 1, i+1) for i in range(len(abc_prices))]
        
        abc_data = MarketData(
            symbol="ABC",
            timeframe="1d",
            data=np.column_stack([abc_prices, abc_prices, abc_prices, abc_prices, np.ones(len(abc_prices)) * 1000]),
            timestamps=abc_timestamps
        )
        
        abc_patterns = self.detect_pattern(abc_data)
        
        if abc_patterns:
            pattern = abc_patterns[0]
            logger.info(f"Stock ABC test: PASSED - Detected pattern with bottoms at indices "
                       f"{pattern.key_points[0][0]} and {pattern.key_points[2][0]}")
        else:
            logger.warning("Stock ABC test: FAILED - No pattern detected")
        
        # Example 2: Stock XYZ (should not confirm without breakout)
        xyz_prices = np.array([100, 98, 95, 97, 99, 101, 103, 100, 98, 96, 94, 96, 98, 100, 102])
        xyz_timestamps = [datetime(2024, 2, i+1) for i in range(len(xyz_prices))]
        
        xyz_data = MarketData(
            symbol="XYZ", 
            timeframe="1d",
            data=np.column_stack([xyz_prices, xyz_prices, xyz_prices, xyz_prices, np.ones(len(xyz_prices)) * 1000]),
            timestamps=xyz_timestamps
        )
        
        xyz_patterns = self.detect_pattern(xyz_data)
        
        if not xyz_patterns:
            logger.info("Stock XYZ test: PASSED - No pattern confirmed (no breakout above 103)")
        else:
            logger.warning("Stock XYZ test: FAILED - Pattern incorrectly detected")
        
        return len(abc_patterns) > 0 and len(xyz_patterns) == 0