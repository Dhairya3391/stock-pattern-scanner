"""
Head and Shoulders Pattern Detection Algorithm.

Implements the exact algorithm from HnS.md reference document following
the methodology from "Technical Analysis of the Financial Markets" by John Murphy.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import logging

from ..core.models import Pattern, PatternConfig, MarketData

logger = logging.getLogger(__name__)


class HeadShouldersDetector:
    """
    Detects Head and Shoulders patterns using the reference algorithm.
    
    Implements the exact steps from HnS.md:
    1. findPeaks - Identify local peaks
    2. identifyHS - Find H&S structure 
    3. drawNeckline - Calculate neckline between troughs
    4. confirmNecklineBreak - Validate breakout
    5. analyzeVolume - Volume confirmation
    6. calculatePriceTarget - Target price calculation
    """
    
    def __init__(self, config: PatternConfig):
        """Initialize detector with configuration."""
        self.config = config
        self.shoulder_tolerance = config.head_shoulders_tolerance
        self.confirmation_lookahead = 10  # Days to look ahead for breakout confirmation
    
    def find_peaks(self, prices: np.ndarray) -> List[int]:
        """
        Find local peaks in price series.
        
        Args:
            prices: Array of price values
            
        Returns:
            List of indices where peaks occur
        """
        peaks = []
        n = len(prices)
        
        # Need at least 3 points to find a peak
        if n < 3:
            return peaks
        
        # Simple local maximum detection: greater than immediate neighbors
        for i in range(1, n - 1):
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                peaks.append(i)
        
        logger.debug(f"Found {len(peaks)} peaks in {n} price points")
        return peaks
    
    def identify_hs(self, prices: np.ndarray, peaks: List[int]) -> List[Tuple[int, int, int]]:
        """
        Identify Head & Shoulders structure from peaks.
        
        Args:
            prices: Price array
            peaks: List of peak indices
            
        Returns:
            List of (left_shoulder, head, right_shoulder) tuples
        """
        hs_patterns = []
        
        # Need at least 3 peaks for H&S pattern
        if len(peaks) < 3:
            return hs_patterns
        
        # Check each consecutive triplet of peaks
        for i in range(len(peaks) - 2):
            ls = peaks[i]      # Left shoulder
            h = peaks[i + 1]   # Head
            rs = peaks[i + 2]  # Right shoulder
            
            # Head must be higher than both shoulders
            if prices[h] > prices[ls] and prices[h] > prices[rs]:
                # Shoulders should be roughly equal (within tolerance)
                shoulder_diff = abs(prices[ls] - prices[rs]) / max(prices[ls], prices[rs])
                
                if shoulder_diff <= self.shoulder_tolerance:
                    hs_patterns.append((ls, h, rs))
                    logger.debug(f"H&S candidate: LS={ls}({prices[ls]:.2f}), H={h}({prices[h]:.2f}), RS={rs}({prices[rs]:.2f})")
        
        return hs_patterns
    
    def find_trough(self, prices: np.ndarray, start: int, end: int) -> int:
        """
        Find index of minimum price between start and end indices.
        
        Args:
            prices: Price array
            start: Start index
            end: End index
            
        Returns:
            Index of minimum price in range
        """
        if start >= end or end >= len(prices):
            return start
        
        trough = start
        for i in range(start + 1, min(end + 1, len(prices))):
            if prices[i] < prices[trough]:
                trough = i
        
        return trough
    
    def draw_neckline(self, prices: np.ndarray, ls: int, h: int, rs: int) -> Tuple[int, int]:
        """
        Draw neckline by finding troughs between shoulders and head.
        
        Args:
            prices: Price array
            ls: Left shoulder index
            h: Head index  
            rs: Right shoulder index
            
        Returns:
            Tuple of (trough1_index, trough2_index)
        """
        # Find trough between left shoulder and head
        t1 = self.find_trough(prices, ls, h)
        
        # Find trough between head and right shoulder
        t2 = self.find_trough(prices, h, rs)
        
        logger.debug(f"Neckline: T1={t1}({prices[t1]:.2f}), T2={t2}({prices[t2]:.2f})")
        return (t1, t2)
    
    def confirm_neckline_break(self, prices: np.ndarray, neckline: Tuple[int, int], 
                              rs: int) -> Tuple[bool, Optional[int]]:
        """
        Confirm neckline break after right shoulder.
        
        Args:
            prices: Price array
            neckline: Tuple of (t1, t2) trough indices
            rs: Right shoulder index
            
        Returns:
            Tuple of (is_confirmed, break_index)
        """
        t1, t2 = neckline
        
        if t1 == t2:  # Avoid division by zero
            return False, None
        
        # Calculate neckline slope
        slope = (prices[t2] - prices[t1]) / (t2 - t1)
        
        # Look for break after right shoulder
        end_idx = min(rs + self.confirmation_lookahead, len(prices))
        
        for i in range(rs + 1, end_idx):
            # Calculate expected neckline price at index i
            neckline_price = prices[t1] + slope * (i - t1)
            
            # For bearish H&S, price breaks below neckline
            if prices[i] < neckline_price:
                logger.debug(f"Neckline break confirmed at index {i}: {prices[i]:.2f} < {neckline_price:.2f}")
                return True, i
        
        return False, None
    
    def analyze_volume(self, volumes: np.ndarray, ls: int, h: int, rs: int, 
                      break_index: int) -> bool:
        """
        Analyze volume trends to validate pattern.
        
        Args:
            volumes: Volume array
            ls: Left shoulder index
            h: Head index
            rs: Right shoulder index
            break_index: Index where neckline break occurred
            
        Returns:
            True if volume criteria are met
        """
        if break_index >= len(volumes):
            return False
        
        head_volume = volumes[h]
        left_volume = volumes[ls]
        right_volume = volumes[rs]
        break_volume = volumes[break_index]
        
        # Volume criteria from reference:
        # 1. Volume highest during head formation
        criteria1 = head_volume > left_volume and head_volume > right_volume
        
        # 2. Volume on right shoulder lower than head
        criteria2 = right_volume < head_volume
        
        # 3. Volume increases on neckline break
        criteria3 = break_volume > right_volume
        
        volume_confirmed = criteria1 and criteria2 and criteria3
        
        logger.debug(f"Volume analysis: Head={head_volume}, Left={left_volume}, "
                    f"Right={right_volume}, Break={break_volume}, Confirmed={volume_confirmed}")
        
        return volume_confirmed
    
    def calculate_price_target(self, prices: np.ndarray, neckline: Tuple[int, int], 
                              h: int, pattern_type: str = "bearish") -> float:
        """
        Calculate price target based on pattern height.
        
        Args:
            prices: Price array
            neckline: Tuple of (t1, t2) trough indices
            h: Head index
            pattern_type: "bearish" or "bullish"
            
        Returns:
            Target price
        """
        t1, t2 = neckline
        
        # Calculate neckline price at head position
        if t1 == t2:
            neckline_at_h = prices[t1]
        else:
            slope = (prices[t2] - prices[t1]) / (t2 - t1)
            neckline_at_h = prices[t1] + slope * (h - t1)
        
        # Pattern height is distance from head to neckline
        pattern_height = prices[h] - neckline_at_h
        
        if pattern_type == "bearish":
            # For bearish H&S, target is below neckline
            target = neckline_at_h - pattern_height
        else:
            # For inverse H&S (bullish), target is above neckline
            target = neckline_at_h + pattern_height
        
        logger.debug(f"Price target calculation: Head={prices[h]:.2f}, "
                    f"Neckline@H={neckline_at_h:.2f}, Height={pattern_height:.2f}, Target={target:.2f}")
        
        return target
    
    def detect_pattern(self, market_data: MarketData) -> List[Pattern]:
        """
        Main detection method that brings together all steps.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            List of detected Head & Shoulders patterns
        """
        patterns = []
        
        try:
            prices = market_data.close_prices
            volumes = market_data.volumes
            timestamps = market_data.timestamps
            
            if len(prices) < 10:  # Need minimum data for pattern
                logger.warning(f"Insufficient data for H&S detection: {len(prices)} points")
                return patterns
            
            # Step 1: Find peaks
            peaks = self.find_peaks(prices)
            
            if len(peaks) < 3:
                logger.debug("Insufficient peaks for H&S pattern")
                return patterns
            
            # Step 2: Identify H&S candidates
            hs_candidates = self.identify_hs(prices, peaks)
            
            # Step 3-6: Process each candidate
            for ls, h, rs in hs_candidates:
                # Draw neckline
                neckline = self.draw_neckline(prices, ls, h, rs)
                
                # Confirm neckline break
                is_confirmed, break_index = self.confirm_neckline_break(prices, neckline, rs)
                
                if is_confirmed and break_index is not None:
                    # Analyze volume
                    volume_confirmed = self.analyze_volume(volumes, ls, h, rs, break_index)
                    
                    # Calculate price target
                    target = self.calculate_price_target(prices, neckline, h, "bearish")
                    
                    # Calculate pattern metrics
                    t1, t2 = neckline
                    neckline_price = (prices[t1] + prices[t2]) / 2  # Approximate neckline level
                    pattern_height = prices[h] - neckline_price
                    
                    # Create pattern object
                    pattern = Pattern(
                        type="Head and Shoulders",
                        symbol=market_data.symbol,
                        timeframe=market_data.timeframe,
                        key_points=[
                            (ls, prices[ls]),  # Left shoulder
                            (h, prices[h]),    # Head
                            (rs, prices[rs]),  # Right shoulder
                            (t1, prices[t1]),  # Left trough
                            (t2, prices[t2])   # Right trough
                        ],
                        confidence=0.8 if volume_confirmed else 0.6,  # Higher confidence with volume
                        traditional_score=0.85,  # High score for confirmed pattern
                        combined_score=0.8 if volume_confirmed else 0.65,
                        entry_price=prices[break_index],
                        target_price=target,
                        stop_loss=prices[h] * 1.02,  # Stop above head with 2% buffer
                        risk_reward_ratio=abs((target - prices[break_index]) / (prices[h] * 1.02 - prices[break_index])),
                        formation_start=timestamps[ls],
                        formation_end=timestamps[rs],
                        breakout_date=timestamps[break_index],
                        status="confirmed",
                        volume_confirmation=volume_confirmed,
                        avg_volume_ratio=volumes[break_index] / np.mean(volumes[max(0, break_index-20):break_index]),
                        pattern_height=pattern_height,
                        duration_days=(timestamps[rs] - timestamps[ls]).days,
                        detection_method="traditional"
                    )
                    
                    patterns.append(pattern)
                    logger.info(f"Detected H&S pattern for {market_data.symbol}: "
                               f"LS={timestamps[ls].date()}, H={timestamps[h].date()}, "
                               f"RS={timestamps[rs].date()}, Target={target:.2f}")
        
        except Exception as e:
            logger.error(f"Error in H&S detection for {market_data.symbol}: {e}")
        
        return patterns