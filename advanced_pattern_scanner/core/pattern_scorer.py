"""
Pattern Scorer for target price calculation and risk assessment.

This module implements comprehensive scoring and risk assessment for detected patterns
using reference methodologies from technical analysis literature.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta

from .models import Pattern, PatternConfig, MarketData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternScorer:
    """
    Comprehensive pattern scoring and risk assessment system.
    
    This class provides methods for:
    1. Target price calculation using multiple methodologies
    2. Risk assessment and stop-loss calculation
    3. Pattern reliability scoring
    4. Risk-reward ratio optimization
    """
    
    def __init__(self, config: PatternConfig):
        """
        Initialize the pattern scorer.
        
        Args:
            config: Pattern detection configuration
        """
        self.config = config
        
        # Scoring weights for different factors
        self.scoring_weights = {
            "pattern_quality": 0.30,      # How well-formed the pattern is
            "volume_confirmation": 0.20,   # Volume validation
            "technical_confluence": 0.20,  # Technical indicator alignment
            "market_context": 0.15,        # Overall market conditions
            "historical_performance": 0.15 # Historical success rate of pattern type
        }
        
        # Historical success rates for different patterns (can be updated with backtesting)
        self.historical_success_rates = {
            "Head and Shoulders": 0.72,
            "Double Bottom": 0.68,
            "Double Top": 0.65,
            "Cup and Handle": 0.78,
            "Ascending Triangle": 0.64,
            "Descending Triangle": 0.62
        }
        
        logger.info("PatternScorer initialized")
    
    def calculate_target_price(self, pattern: Pattern, market_data: MarketData, 
                             method: str = "measured_move") -> float:
        """
        Calculate target price using specified methodology.
        
        Args:
            pattern: Pattern to calculate target for
            market_data: Market data for analysis
            method: Calculation method ("measured_move", "fibonacci", "support_resistance")
            
        Returns:
            Calculated target price
        """
        if method == "measured_move":
            return self._calculate_measured_move_target(pattern, market_data)
        elif method == "fibonacci":
            return self._calculate_fibonacci_target(pattern, market_data)
        elif method == "support_resistance":
            return self._calculate_support_resistance_target(pattern, market_data)
        else:
            logger.warning(f"Unknown target calculation method: {method}, using measured_move")
            return self._calculate_measured_move_target(pattern, market_data)
    
    def _calculate_measured_move_target(self, pattern: Pattern, market_data: MarketData) -> float:
        """
        Calculate target using measured move technique.
        
        Args:
            pattern: Pattern to analyze
            market_data: Market data
            
        Returns:
            Target price using measured move
        """
        if not pattern.key_points or len(pattern.key_points) < 2:
            return pattern.entry_price * 1.1  # Default 10% target
        
        if pattern.type == "Head and Shoulders":
            # For H&S: Target = Neckline - (Head - Neckline)
            if len(pattern.key_points) >= 5:
                head_price = pattern.key_points[1][1]  # Head
                neckline_price = (pattern.key_points[3][1] + pattern.key_points[4][1]) / 2  # Average of troughs
                pattern_height = head_price - neckline_price
                target = neckline_price - pattern_height
            else:
                target = pattern.entry_price * 0.9  # Default bearish target
        
        elif pattern.type in ["Double Bottom", "Double Top"]:
            # For double patterns: Target = Breakout + Pattern Height
            if len(pattern.key_points) >= 3:
                if pattern.type == "Double Bottom":
                    resistance = pattern.key_points[1][1]  # Peak between bottoms
                    support = min(pattern.key_points[0][1], pattern.key_points[2][1])  # Lower of two bottoms
                    pattern_height = resistance - support
                    target = pattern.entry_price + pattern_height
                else:  # Double Top
                    resistance = max(pattern.key_points[0][1], pattern.key_points[2][1])  # Higher of two tops
                    support = pattern.key_points[1][1]  # Trough between tops
                    pattern_height = resistance - support
                    target = pattern.entry_price - pattern_height
            else:
                multiplier = 1.15 if pattern.type == "Double Bottom" else 0.85
                target = pattern.entry_price * multiplier
        
        elif pattern.type == "Cup and Handle":
            # For Cup & Handle: Target = Breakout + Cup Depth
            if len(pattern.key_points) >= 3:
                cup_rim = pattern.key_points[0][1]  # Left rim
                cup_bottom = pattern.key_points[1][1]  # Cup bottom
                cup_depth = cup_rim - cup_bottom
                target = pattern.entry_price + cup_depth
            else:
                target = pattern.entry_price * 1.2  # Default bullish target
        
        else:
            # Generic calculation for other patterns
            if pattern.is_bullish:
                target = pattern.entry_price * 1.15
            else:
                target = pattern.entry_price * 0.85
        
        logger.debug(f"Measured move target for {pattern.type}: {target:.2f}")
        return target
    
    def _calculate_fibonacci_target(self, pattern: Pattern, market_data: MarketData) -> float:
        """
        Calculate target using Fibonacci extensions.
        
        Args:
            pattern: Pattern to analyze
            market_data: Market data
            
        Returns:
            Target price using Fibonacci extensions
        """
        if not pattern.key_points or len(pattern.key_points) < 2:
            return self._calculate_measured_move_target(pattern, market_data)
        
        # Common Fibonacci extension levels
        fib_levels = [1.272, 1.414, 1.618, 2.0]
        
        # Calculate pattern range
        prices = [point[1] for point in pattern.key_points]
        pattern_high = max(prices)
        pattern_low = min(prices)
        pattern_range = pattern_high - pattern_low
        
        if pattern.is_bullish:
            # For bullish patterns, extend upward from breakout
            base_target = self._calculate_measured_move_target(pattern, market_data)
            fib_extension = pattern_range * fib_levels[1]  # Use 1.414 level
            target = pattern.entry_price + fib_extension
        else:
            # For bearish patterns, extend downward from breakout
            base_target = self._calculate_measured_move_target(pattern, market_data)
            fib_extension = pattern_range * fib_levels[1]  # Use 1.414 level
            target = pattern.entry_price - fib_extension
        
        logger.debug(f"Fibonacci target for {pattern.type}: {target:.2f}")
        return target
    
    def _calculate_support_resistance_target(self, pattern: Pattern, market_data: MarketData) -> float:
        """
        Calculate target based on support/resistance levels.
        
        Args:
            pattern: Pattern to analyze
            market_data: Market data
            
        Returns:
            Target price based on S/R levels
        """
        prices = market_data.close_prices
        
        # Find significant support/resistance levels
        sr_levels = self._find_support_resistance_levels(prices)
        
        if not sr_levels:
            return self._calculate_measured_move_target(pattern, market_data)
        
        # Find next significant level in the direction of the pattern
        if pattern.is_bullish:
            # Find next resistance above entry price
            resistance_levels = [level for level in sr_levels if level > pattern.entry_price]
            target = min(resistance_levels) if resistance_levels else pattern.entry_price * 1.1
        else:
            # Find next support below entry price
            support_levels = [level for level in sr_levels if level < pattern.entry_price]
            target = max(support_levels) if support_levels else pattern.entry_price * 0.9
        
        logger.debug(f"S/R target for {pattern.type}: {target:.2f}")
        return target
    
    def _find_support_resistance_levels(self, prices: np.ndarray, 
                                      min_touches: int = 2, tolerance: float = 0.02) -> List[float]:
        """
        Find significant support and resistance levels.
        
        Args:
            prices: Price array
            min_touches: Minimum number of touches to confirm level
            tolerance: Price tolerance for level confirmation (2%)
            
        Returns:
            List of significant S/R levels
        """
        from scipy.signal import find_peaks
        
        # Find peaks and troughs
        peaks, _ = find_peaks(prices, distance=5)
        troughs, _ = find_peaks(-prices, distance=5)
        
        # Combine all potential levels
        all_levels = list(prices[peaks]) + list(prices[troughs])
        
        # Group similar levels
        sr_levels = []
        for level in all_levels:
            # Count how many times this level was touched
            touches = 0
            for price in all_levels:
                if abs(price - level) / level <= tolerance:
                    touches += 1
            
            if touches >= min_touches and level not in sr_levels:
                # Check if similar level already exists
                similar_exists = any(abs(existing - level) / existing <= tolerance 
                                   for existing in sr_levels)
                if not similar_exists:
                    sr_levels.append(level)
        
        return sorted(sr_levels)
    
    def calculate_stop_loss(self, pattern: Pattern, market_data: MarketData, 
                          method: str = "pattern_based") -> float:
        """
        Calculate stop-loss level using specified method.
        
        Args:
            pattern: Pattern to calculate stop-loss for
            market_data: Market data for analysis
            method: Calculation method ("pattern_based", "atr", "percentage")
            
        Returns:
            Stop-loss price
        """
        if method == "pattern_based":
            return self._calculate_pattern_based_stop(pattern, market_data)
        elif method == "atr":
            return self._calculate_atr_based_stop(pattern, market_data)
        elif method == "percentage":
            return self._calculate_percentage_stop(pattern, market_data)
        else:
            logger.warning(f"Unknown stop-loss method: {method}, using pattern_based")
            return self._calculate_pattern_based_stop(pattern, market_data)
    
    def _calculate_pattern_based_stop(self, pattern: Pattern, market_data: MarketData) -> float:
        """
        Calculate stop-loss based on pattern structure.
        
        Args:
            pattern: Pattern to analyze
            market_data: Market data
            
        Returns:
            Pattern-based stop-loss level
        """
        if pattern.type == "Head and Shoulders":
            # Stop above the head with small buffer
            if len(pattern.key_points) >= 2:
                head_price = pattern.key_points[1][1]  # Head
                stop_loss = head_price * 1.02  # 2% buffer above head
            else:
                stop_loss = pattern.entry_price * 1.05
        
        elif pattern.type == "Double Bottom":
            # Stop below the lower of the two bottoms
            if len(pattern.key_points) >= 3:
                lower_bottom = min(pattern.key_points[0][1], pattern.key_points[2][1])
                stop_loss = lower_bottom * 0.98  # 2% buffer below
            else:
                stop_loss = pattern.entry_price * 0.95
        
        elif pattern.type == "Double Top":
            # Stop above the higher of the two tops
            if len(pattern.key_points) >= 3:
                higher_top = max(pattern.key_points[0][1], pattern.key_points[2][1])
                stop_loss = higher_top * 1.02  # 2% buffer above
            else:
                stop_loss = pattern.entry_price * 1.05
        
        elif pattern.type == "Cup and Handle":
            # Stop below the handle low
            if len(pattern.key_points) >= 4:
                handle_low = pattern.key_points[3][1]  # Handle low
                stop_loss = handle_low * 0.98  # 2% buffer below
            else:
                stop_loss = pattern.entry_price * 0.95
        
        else:
            # Generic stop-loss calculation
            if pattern.is_bullish:
                stop_loss = pattern.entry_price * 0.95  # 5% below entry
            else:
                stop_loss = pattern.entry_price * 1.05  # 5% above entry
        
        logger.debug(f"Pattern-based stop for {pattern.type}: {stop_loss:.2f}")
        return stop_loss
    
    def _calculate_atr_based_stop(self, pattern: Pattern, market_data: MarketData, 
                                atr_multiplier: float = 2.0) -> float:
        """
        Calculate stop-loss based on Average True Range (ATR).
        
        Args:
            pattern: Pattern to analyze
            market_data: Market data
            atr_multiplier: Multiplier for ATR (default 2.0)
            
        Returns:
            ATR-based stop-loss level
        """
        # Calculate ATR
        atr = self._calculate_atr(market_data, period=14)
        
        if pattern.is_bullish:
            stop_loss = pattern.entry_price - (atr * atr_multiplier)
        else:
            stop_loss = pattern.entry_price + (atr * atr_multiplier)
        
        logger.debug(f"ATR-based stop for {pattern.type}: {stop_loss:.2f} (ATR: {atr:.2f})")
        return stop_loss
    
    def _calculate_percentage_stop(self, pattern: Pattern, market_data: MarketData, 
                                 stop_percentage: float = 0.05) -> float:
        """
        Calculate stop-loss based on fixed percentage.
        
        Args:
            pattern: Pattern to analyze
            market_data: Market data
            stop_percentage: Stop-loss percentage (default 5%)
            
        Returns:
            Percentage-based stop-loss level
        """
        if pattern.is_bullish:
            stop_loss = pattern.entry_price * (1 - stop_percentage)
        else:
            stop_loss = pattern.entry_price * (1 + stop_percentage)
        
        logger.debug(f"Percentage stop for {pattern.type}: {stop_loss:.2f} ({stop_percentage*100}%)")
        return stop_loss
    
    def _calculate_atr(self, market_data: MarketData, period: int = 14) -> float:
        """
        Calculate Average True Range.
        
        Args:
            market_data: Market data
            period: ATR calculation period
            
        Returns:
            ATR value
        """
        if len(market_data.data) < period + 1:
            return 0.0
        
        highs = market_data.high_prices
        lows = market_data.low_prices
        closes = market_data.close_prices
        
        # Calculate True Range
        tr_list = []
        for i in range(1, len(market_data.data)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr = max(tr1, tr2, tr3)
            tr_list.append(tr)
        
        # Calculate ATR as simple moving average of TR
        if len(tr_list) >= period:
            atr = np.mean(tr_list[-period:])
        else:
            atr = np.mean(tr_list)
        
        return atr
    
    def score_pattern_quality(self, pattern: Pattern, market_data: MarketData) -> float:
        """
        Score pattern quality based on multiple factors.
        
        Args:
            pattern: Pattern to score
            market_data: Market data for analysis
            
        Returns:
            Quality score (0-1)
        """
        scores = {}
        
        # 1. Pattern formation quality
        scores["formation"] = self._score_pattern_formation(pattern)
        
        # 2. Volume confirmation
        scores["volume"] = self._score_volume_confirmation(pattern, market_data)
        
        # 3. Technical confluence
        scores["technical"] = self._score_technical_confluence(pattern, market_data)
        
        # 4. Market context
        scores["market_context"] = self._score_market_context(pattern, market_data)
        
        # 5. Historical performance
        scores["historical"] = self._score_historical_performance(pattern)
        
        # Calculate weighted score
        total_score = sum(scores[factor] * self.scoring_weights[factor.replace("formation", "pattern_quality")]
                         for factor in scores)
        
        logger.debug(f"Pattern quality scores for {pattern.type}: {scores}, Total: {total_score:.3f}")
        return min(max(total_score, 0.0), 1.0)
    
    def _score_pattern_formation(self, pattern: Pattern) -> float:
        """Score the quality of pattern formation."""
        score = 0.5  # Base score
        
        # Duration scoring
        if hasattr(pattern, 'duration_days'):
            if self.config.min_pattern_duration <= pattern.duration_days <= self.config.max_pattern_duration:
                score += 0.2
            elif pattern.duration_days < self.config.min_pattern_duration:
                score -= 0.1
        
        # Pattern height scoring
        if hasattr(pattern, 'pattern_height') and pattern.pattern_height > 0:
            height_ratio = pattern.pattern_height / pattern.entry_price
            if height_ratio >= 0.1:  # Significant pattern height
                score += 0.2
            elif height_ratio >= 0.05:
                score += 0.1
        
        # Key points validation
        if len(pattern.key_points) >= 3:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _score_volume_confirmation(self, pattern: Pattern, market_data: MarketData) -> float:
        """Score volume confirmation quality."""
        if pattern.volume_confirmation:
            # Strong volume confirmation
            if pattern.avg_volume_ratio >= 2.0:
                return 1.0
            elif pattern.avg_volume_ratio >= 1.5:
                return 0.8
            elif pattern.avg_volume_ratio >= 1.2:
                return 0.6
            else:
                return 0.4
        else:
            # No volume confirmation
            return 0.2
    
    def _score_technical_confluence(self, pattern: Pattern, market_data: MarketData) -> float:
        """Score technical indicator confluence."""
        # Simplified technical scoring - can be enhanced with actual indicators
        score = 0.5  # Base score
        
        # Check if pattern aligns with trend
        if len(market_data.close_prices) >= 50:
            recent_trend = (market_data.close_prices[-1] - market_data.close_prices[-50]) / market_data.close_prices[-50]
            
            if pattern.is_bullish and recent_trend > 0:
                score += 0.3  # Bullish pattern in uptrend
            elif pattern.is_bearish and recent_trend < 0:
                score += 0.3  # Bearish pattern in downtrend
            elif pattern.is_bullish and recent_trend < -0.1:
                score += 0.2  # Bullish reversal pattern
            elif pattern.is_bearish and recent_trend > 0.1:
                score += 0.2  # Bearish reversal pattern
        
        return min(max(score, 0.0), 1.0)
    
    def _score_market_context(self, pattern: Pattern, market_data: MarketData) -> float:
        """Score market context and conditions."""
        # Simplified market context scoring
        score = 0.5  # Base score
        
        # Volatility assessment
        if len(market_data.close_prices) >= 20:
            recent_volatility = np.std(market_data.close_prices[-20:]) / np.mean(market_data.close_prices[-20:])
            
            if 0.02 <= recent_volatility <= 0.05:  # Moderate volatility is good
                score += 0.3
            elif recent_volatility > 0.08:  # High volatility reduces reliability
                score -= 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _score_historical_performance(self, pattern: Pattern) -> float:
        """Score based on historical pattern performance."""
        return self.historical_success_rates.get(pattern.type, 0.6)
    
    def calculate_risk_reward_ratio(self, pattern: Pattern, target_price: float, 
                                  stop_loss: float) -> float:
        """
        Calculate risk-reward ratio for the pattern.
        
        Args:
            pattern: Pattern to analyze
            target_price: Target price
            stop_loss: Stop-loss price
            
        Returns:
            Risk-reward ratio
        """
        if pattern.entry_price == stop_loss:
            return 0.0
        
        potential_reward = abs(target_price - pattern.entry_price)
        potential_risk = abs(pattern.entry_price - stop_loss)
        
        if potential_risk == 0:
            return float('inf')
        
        risk_reward_ratio = potential_reward / potential_risk
        
        logger.debug(f"Risk-reward ratio for {pattern.type}: {risk_reward_ratio:.2f} "
                    f"(Reward: {potential_reward:.2f}, Risk: {potential_risk:.2f})")
        
        return risk_reward_ratio
    
    def optimize_position_size(self, pattern: Pattern, account_balance: float, 
                             max_risk_percent: float = None) -> Dict[str, float]:
        """
        Calculate optimal position size based on risk management.
        
        Args:
            pattern: Pattern to analyze
            account_balance: Total account balance
            max_risk_percent: Maximum risk percentage (uses config if None)
            
        Returns:
            Dictionary with position sizing information
        """
        if max_risk_percent is None:
            max_risk_percent = self.config.max_risk_percent / 100
        
        # Calculate maximum dollar risk
        max_dollar_risk = account_balance * max_risk_percent
        
        # Calculate risk per share
        risk_per_share = abs(pattern.entry_price - pattern.stop_loss)
        
        if risk_per_share == 0:
            return {"shares": 0, "position_value": 0, "risk_amount": 0}
        
        # Calculate position size
        max_shares = int(max_dollar_risk / risk_per_share)
        position_value = max_shares * pattern.entry_price
        actual_risk = max_shares * risk_per_share
        
        # Calculate potential profit
        potential_profit = max_shares * abs(pattern.target_price - pattern.entry_price)
        
        position_info = {
            "shares": max_shares,
            "position_value": position_value,
            "risk_amount": actual_risk,
            "potential_profit": potential_profit,
            "risk_percent": (actual_risk / account_balance) * 100,
            "position_percent": (position_value / account_balance) * 100
        }
        
        logger.debug(f"Position sizing for {pattern.type}: {max_shares} shares, "
                    f"${position_value:.2f} position, ${actual_risk:.2f} risk")
        
        return position_info