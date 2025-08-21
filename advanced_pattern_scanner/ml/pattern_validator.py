"""
Pattern Validator for Advanced Pattern Scanner.

This module provides ML-based validation for patterns detected by traditional
algorithms, combining rule-based detection with machine learning confirmation
to reduce false positives and improve accuracy.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta

from ..core.models import Pattern, PatternConfig, MarketData
from .model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternValidator:
    """
    Validates patterns detected by traditional algorithms using ML models.
    
    This class provides a hybrid validation system that combines traditional
    technical analysis with machine learning to improve pattern detection
    accuracy and reduce false positives.
    """
    
    def __init__(self, config: PatternConfig, model_manager: ModelManager):
        """
        Initialize the pattern validator.
        
        Args:
            config: Pattern detection configuration
            model_manager: ML model manager instance
        """
        self.config = config
        self.model_manager = model_manager
        
        # Validation thresholds for different pattern types
        self.pattern_thresholds = {
            "Head and Shoulders": 0.75,
            "Double Bottom": 0.70,
            "Double Top": 0.70,
            "Cup and Handle": 0.80,
            "Ascending Triangle": 0.65,
            "Descending Triangle": 0.65,
            "Symmetrical Triangle": 0.60
        }
        
        # Pattern-specific validation weights
        self.validation_weights = {
            "ml_confidence": 0.4,      # ML model confidence
            "traditional_score": 0.3,  # Traditional algorithm score
            "volume_confirmation": 0.2, # Volume analysis
            "technical_indicators": 0.1 # Technical indicator confirmation
        }
        
        logger.info("PatternValidator initialized")
    
    def validate_pattern(self, pattern: Pattern, market_data: MarketData) -> Tuple[bool, float, Dict]:
        """
        Validate a pattern using ML models and additional checks.
        
        Args:
            pattern: Pattern to validate
            market_data: Market data for validation
            
        Returns:
            Tuple of (is_valid, confidence_score, validation_details)
        """
        validation_details = {
            "ml_validation": {},
            "volume_validation": {},
            "technical_validation": {},
            "pattern_specific": {},
            "final_score": 0.0
        }
        
        try:
            # ML-based validation
            ml_confidence = self.model_manager.validate_pattern(pattern, market_data)
            validation_details["ml_validation"] = {
                "confidence": ml_confidence,
                "passed": ml_confidence >= self.config.min_confidence
            }
            
            # Volume validation
            volume_score, volume_details = self._validate_volume(pattern, market_data)
            validation_details["volume_validation"] = volume_details
            
            # Technical indicator validation
            tech_score, tech_details = self._validate_technical_indicators(pattern, market_data)
            validation_details["technical_validation"] = tech_details
            
            # Pattern-specific validation
            pattern_score, pattern_details = self._validate_pattern_specific(pattern, market_data)
            validation_details["pattern_specific"] = pattern_details
            
            # Calculate weighted final score
            final_score = (
                ml_confidence * self.validation_weights["ml_confidence"] +
                pattern.traditional_score * self.validation_weights["traditional_score"] +
                volume_score * self.validation_weights["volume_confirmation"] +
                tech_score * self.validation_weights["technical_indicators"]
            )
            
            validation_details["final_score"] = final_score
            
            # Determine if pattern is valid
            threshold = self.pattern_thresholds.get(pattern.type, self.config.min_combined_score)
            is_valid = final_score >= threshold
            
            # Update pattern with validation results
            pattern.confidence = ml_confidence
            pattern.combined_score = final_score
            
            logger.debug(f"Pattern validation: {pattern.type} - Score: {final_score:.3f}, Valid: {is_valid}")
            
            return is_valid, final_score, validation_details
            
        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            return False, 0.0, validation_details
    
    def _validate_volume(self, pattern: Pattern, market_data: MarketData) -> Tuple[float, Dict]:
        """
        Validate pattern based on volume analysis.
        
        Args:
            pattern: Pattern to validate
            market_data: Market data
            
        Returns:
            Tuple of (volume_score, validation_details)
        """
        details = {
            "avg_volume_ratio": 0.0,
            "volume_trend": "neutral",
            "breakout_volume": 0.0,
            "score": 0.0
        }
        
        try:
            volumes = market_data.volumes
            if len(volumes) < self.config.volume_lookback_days:
                details["score"] = 0.5  # Neutral score for insufficient data
                return 0.5, details
            
            # Calculate average volume ratio
            recent_volume = np.mean(volumes[-5:])  # Last 5 periods
            baseline_volume = np.mean(volumes[-self.config.volume_lookback_days:-5])
            
            if baseline_volume > 0:
                volume_ratio = recent_volume / baseline_volume
                details["avg_volume_ratio"] = volume_ratio
                pattern.avg_volume_ratio = volume_ratio
            else:
                volume_ratio = 1.0
                details["avg_volume_ratio"] = 1.0
            
            # Analyze volume trend
            if len(volumes) >= 10:
                volume_trend = np.polyfit(range(len(volumes[-10:])), volumes[-10:], 1)[0]
                if volume_trend > 0:
                    details["volume_trend"] = "increasing"
                elif volume_trend < 0:
                    details["volume_trend"] = "decreasing"
                else:
                    details["volume_trend"] = "neutral"
            
            # Check for breakout volume (if pattern has breakout)
            if pattern.breakout_date and len(volumes) > 0:
                details["breakout_volume"] = volumes[-1]
            
            # Calculate volume score
            score = 0.0
            
            # Volume ratio component (40% of volume score)
            if volume_ratio >= self.config.min_volume_ratio:
                score += 0.4 * min(1.0, volume_ratio / 2.0)
            
            # Volume trend component (30% of volume score)
            if pattern.is_bullish and details["volume_trend"] == "increasing":
                score += 0.3
            elif pattern.is_bearish and details["volume_trend"] == "decreasing":
                score += 0.3
            elif details["volume_trend"] == "neutral":
                score += 0.15
            
            # Volume confirmation component (30% of volume score)
            if pattern.volume_confirmation:
                score += 0.3
            
            details["score"] = score
            return score, details
            
        except Exception as e:
            logger.warning(f"Volume validation failed: {e}")
            details["score"] = 0.5
            return 0.5, details
    
    def _validate_technical_indicators(self, pattern: Pattern, 
                                     market_data: MarketData) -> Tuple[float, Dict]:
        """
        Validate pattern using technical indicators.
        
        Args:
            pattern: Pattern to validate
            market_data: Market data
            
        Returns:
            Tuple of (technical_score, validation_details)
        """
        details = {
            "rsi_signal": "neutral",
            "macd_signal": "neutral",
            "trend_alignment": "neutral",
            "score": 0.0
        }
        
        try:
            closes = market_data.close_prices
            if len(closes) < 50:  # Need sufficient data for indicators
                details["score"] = 0.5
                return 0.5, details
            
            # Calculate RSI
            rsi = self._calculate_rsi(closes, self.config.rsi_period)
            if len(rsi) > 0:
                current_rsi = rsi[-1]
                
                if pattern.is_bullish:
                    if current_rsi < 70:  # Not overbought
                        details["rsi_signal"] = "bullish"
                    elif current_rsi > 80:
                        details["rsi_signal"] = "bearish"
                elif pattern.is_bearish:
                    if current_rsi > 30:  # Not oversold
                        details["rsi_signal"] = "bearish"
                    elif current_rsi < 20:
                        details["rsi_signal"] = "bullish"
            
            # Calculate MACD
            macd_line, signal_line = self._calculate_macd(closes)
            if len(macd_line) > 0 and len(signal_line) > 0:
                macd_diff = macd_line[-1] - signal_line[-1]
                
                if pattern.is_bullish and macd_diff > 0:
                    details["macd_signal"] = "bullish"
                elif pattern.is_bearish and macd_diff < 0:
                    details["macd_signal"] = "bearish"
            
            # Check trend alignment
            if len(closes) >= 50:
                sma_20 = np.mean(closes[-20:])
                sma_50 = np.mean(closes[-50:])
                
                if pattern.is_bullish and sma_20 > sma_50:
                    details["trend_alignment"] = "bullish"
                elif pattern.is_bearish and sma_20 < sma_50:
                    details["trend_alignment"] = "bearish"
            
            # Calculate technical score
            score = 0.0
            
            # RSI component (40%)
            if details["rsi_signal"] == "bullish" and pattern.is_bullish:
                score += 0.4
            elif details["rsi_signal"] == "bearish" and pattern.is_bearish:
                score += 0.4
            elif details["rsi_signal"] == "neutral":
                score += 0.2
            
            # MACD component (35%)
            if details["macd_signal"] == "bullish" and pattern.is_bullish:
                score += 0.35
            elif details["macd_signal"] == "bearish" and pattern.is_bearish:
                score += 0.35
            elif details["macd_signal"] == "neutral":
                score += 0.175
            
            # Trend alignment component (25%)
            if details["trend_alignment"] == "bullish" and pattern.is_bullish:
                score += 0.25
            elif details["trend_alignment"] == "bearish" and pattern.is_bearish:
                score += 0.25
            elif details["trend_alignment"] == "neutral":
                score += 0.125
            
            details["score"] = score
            return score, details
            
        except Exception as e:
            logger.warning(f"Technical indicator validation failed: {e}")
            details["score"] = 0.5
            return 0.5, details
    
    def _validate_pattern_specific(self, pattern: Pattern, 
                                 market_data: MarketData) -> Tuple[float, Dict]:
        """
        Perform pattern-specific validation checks.
        
        Args:
            pattern: Pattern to validate
            market_data: Market data
            
        Returns:
            Tuple of (pattern_score, validation_details)
        """
        details = {
            "duration_check": False,
            "height_check": False,
            "symmetry_check": False,
            "formation_quality": 0.0,
            "score": 0.0
        }
        
        try:
            # Duration validation
            if (self.config.min_pattern_duration <= 
                pattern.duration_days <= 
                self.config.max_pattern_duration):
                details["duration_check"] = True
            
            # Height validation (pattern should be significant)
            closes = market_data.close_prices
            if len(closes) > 0:
                price_range = np.max(closes) - np.min(closes)
                avg_price = np.mean(closes)
                height_ratio = price_range / avg_price if avg_price > 0 else 0
                
                if height_ratio >= 0.05:  # At least 5% price movement
                    details["height_check"] = True
            
            # Pattern-specific checks
            if pattern.type == "Head and Shoulders":
                details.update(self._validate_head_shoulders(pattern, market_data))
            elif pattern.type in ["Double Bottom", "Double Top"]:
                details.update(self._validate_double_pattern(pattern, market_data))
            elif pattern.type == "Cup and Handle":
                details.update(self._validate_cup_handle(pattern, market_data))
            
            # Calculate pattern-specific score
            score = 0.0
            
            if details["duration_check"]:
                score += 0.3
            if details["height_check"]:
                score += 0.3
            if details["symmetry_check"]:
                score += 0.2
            
            score += details["formation_quality"] * 0.2
            
            details["score"] = score
            return score, details
            
        except Exception as e:
            logger.warning(f"Pattern-specific validation failed: {e}")
            details["score"] = 0.5
            return 0.5, details
    
    def _validate_head_shoulders(self, pattern: Pattern, 
                               market_data: MarketData) -> Dict:
        """Validate Head and Shoulders pattern specifics."""
        details = {"symmetry_check": False, "formation_quality": 0.0}
        
        try:
            if len(pattern.key_points) >= 5:  # Head, shoulders, and neckline points
                # Check shoulder symmetry
                left_shoulder = pattern.key_points[0][1]
                head = pattern.key_points[2][1]
                right_shoulder = pattern.key_points[4][1]
                
                shoulder_diff = abs(left_shoulder - right_shoulder)
                head_height = abs(head - (left_shoulder + right_shoulder) / 2)
                
                if head_height > 0:
                    symmetry_ratio = shoulder_diff / head_height
                    if symmetry_ratio <= self.config.head_shoulders_tolerance:
                        details["symmetry_check"] = True
                        details["formation_quality"] = 1.0 - symmetry_ratio
            
        except Exception as e:
            logger.warning(f"Head and Shoulders validation failed: {e}")
        
        return details
    
    def _validate_double_pattern(self, pattern: Pattern, 
                               market_data: MarketData) -> Dict:
        """Validate Double Top/Bottom pattern specifics."""
        details = {"symmetry_check": False, "formation_quality": 0.0}
        
        try:
            if len(pattern.key_points) >= 2:
                # Check peak/trough similarity
                first_extreme = pattern.key_points[0][1]
                second_extreme = pattern.key_points[1][1]
                
                price_diff = abs(first_extreme - second_extreme)
                avg_price = (first_extreme + second_extreme) / 2
                
                if avg_price > 0:
                    diff_ratio = price_diff / avg_price
                    if diff_ratio <= self.config.double_pattern_tolerance:
                        details["symmetry_check"] = True
                        details["formation_quality"] = 1.0 - diff_ratio
            
        except Exception as e:
            logger.warning(f"Double pattern validation failed: {e}")
        
        return details
    
    def _validate_cup_handle(self, pattern: Pattern, 
                           market_data: MarketData) -> Dict:
        """Validate Cup and Handle pattern specifics."""
        details = {"symmetry_check": False, "formation_quality": 0.0}
        
        try:
            closes = market_data.close_prices
            if len(closes) > 0 and pattern.pattern_height > 0:
                avg_price = np.mean(closes)
                depth_ratio = pattern.pattern_height / avg_price
                
                # Check cup depth
                if (self.config.cup_handle_depth_min <= 
                    depth_ratio <= 
                    self.config.cup_handle_depth_max):
                    details["symmetry_check"] = True
                    
                    # Quality based on ideal depth (around 30%)
                    ideal_depth = 0.30
                    depth_quality = 1.0 - abs(depth_ratio - ideal_depth) / ideal_depth
                    details["formation_quality"] = max(0.0, depth_quality)
            
        except Exception as e:
            logger.warning(f"Cup and Handle validation failed: {e}")
        
        return details
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return np.array([])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        if len(prices) < max(self.config.macd_slow, self.config.macd_signal):
            return np.array([]), np.array([])
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, self.config.macd_fast)
        ema_slow = self._calculate_ema(prices, self.config.macd_slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = self._calculate_ema(macd_line, self.config.macd_signal)
        
        return macd_line, signal_line
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def batch_validate(self, patterns: List[Pattern], 
                      market_data_list: List[MarketData]) -> List[Tuple[bool, float, Dict]]:
        """
        Validate multiple patterns in batch.
        
        Args:
            patterns: List of patterns to validate
            market_data_list: List of corresponding market data
            
        Returns:
            List of validation results
        """
        results = []
        
        for pattern, market_data in zip(patterns, market_data_list):
            try:
                result = self.validate_pattern(pattern, market_data)
                results.append(result)
            except Exception as e:
                logger.warning(f"Batch validation failed for pattern: {e}")
                results.append((False, 0.0, {}))
        
        return results
    
    def get_validation_summary(self, validation_results: List[Tuple[bool, float, Dict]]) -> Dict:
        """
        Generate summary statistics for validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Summary statistics dictionary
        """
        if not validation_results:
            return {}
        
        valid_count = sum(1 for result in validation_results if result[0])
        total_count = len(validation_results)
        scores = [result[1] for result in validation_results]
        
        summary = {
            "total_patterns": total_count,
            "valid_patterns": valid_count,
            "validation_rate": valid_count / total_count if total_count > 0 else 0.0,
            "average_score": np.mean(scores) if scores else 0.0,
            "median_score": np.median(scores) if scores else 0.0,
            "min_score": np.min(scores) if scores else 0.0,
            "max_score": np.max(scores) if scores else 0.0
        }
        
        return summary