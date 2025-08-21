"""
Pattern Detection Engine - Main orchestrator for all pattern detectors and ML validation.

This module provides the main detection engine that coordinates all pattern detectors,
applies ML validation, and manages the complete pattern detection workflow.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .models import Pattern, PatternConfig, MarketData
from .data_manager import DataManager
from ..patterns.head_shoulders import HeadShouldersDetector
from ..patterns.double_bottom import DoubleBottomDetector
from ..patterns.cup_handle import CupHandleDetector
from ..ml.hybrid_validator import HybridValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternEngine:
    """
    Main pattern detection engine that orchestrates all detectors and validation.
    
    This class coordinates the entire pattern detection workflow:
    1. Data preprocessing and validation
    2. Pattern detection using multiple algorithms
    3. ML-based validation and scoring
    4. Result aggregation and ranking
    """
    
    def __init__(self, config: PatternConfig):
        """
        Initialize the pattern detection engine.
        
        Args:
            config: Pattern detection configuration
        """
        self.config = config
        self.data_manager = DataManager(config)
        
        # Initialize pattern detectors
        self.detectors = {
            "Head and Shoulders": HeadShouldersDetector(config),
            "Double Bottom": DoubleBottomDetector(config),
            "Cup and Handle": CupHandleDetector(config)
        }
        
        # Initialize hybrid validator
        self.hybrid_validator = HybridValidator(config)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_concurrent_requests if config.enable_parallel_processing else 1
        )
        
        # Detection statistics
        self.stats = {
            "total_scans": 0,
            "successful_scans": 0,
            "patterns_detected": 0,
            "patterns_validated": 0,
            "detection_times": [],
            "validation_times": []
        }
        
        logger.info(f"PatternEngine initialized with {len(self.detectors)} detectors")
    
    def detect_patterns_single_stock(self, symbol: str, period: str = "1y", 
                                   pattern_types: Optional[List[str]] = None) -> Dict[str, List[Pattern]]:
        """
        Detect patterns for a single stock.
        
        Args:
            symbol: Stock symbol to analyze
            period: Time period for data ('1y', '6mo', etc.)
            pattern_types: List of pattern types to detect (None for all)
            
        Returns:
            Dictionary mapping pattern types to lists of detected patterns
        """
        start_time = datetime.now()
        results = {}
        
        try:
            # Fetch and preprocess data
            df = self.data_manager.fetch_stock_data(symbol, period)
            if df is None or df.empty:
                logger.warning(f"No data available for {symbol}")
                return results
            
            market_data = self.data_manager.preprocess_for_analysis(df, symbol)
            if market_data is None:
                logger.warning(f"Failed to preprocess data for {symbol}")
                return results
            
            # Determine which patterns to detect
            if pattern_types is None:
                pattern_types = list(self.detectors.keys())
            
            # Detect patterns using each detector
            all_patterns = []
            for pattern_type in pattern_types:
                if pattern_type in self.detectors:
                    detector = self.detectors[pattern_type]
                    patterns = detector.detect_pattern(market_data)
                    
                    if patterns:
                        results[pattern_type] = patterns
                        all_patterns.extend(patterns)
                        logger.info(f"Detected {len(patterns)} {pattern_type} patterns for {symbol}")
            
            # Apply hybrid validation to all detected patterns
            if all_patterns:
                validated_patterns = self._validate_patterns(all_patterns, market_data)
                
                # Update results with validated patterns
                results = self._organize_validated_patterns(validated_patterns)
                
                self.stats["patterns_detected"] += len(all_patterns)
                self.stats["patterns_validated"] += len(validated_patterns)
            
            self.stats["successful_scans"] += 1
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
        
        finally:
            self.stats["total_scans"] += 1
            detection_time = (datetime.now() - start_time).total_seconds()
            self.stats["detection_times"].append(detection_time)
            
            logger.debug(f"Pattern detection for {symbol} completed in {detection_time:.2f}s")
        
        return results
    
    def detect_patterns_batch(self, symbols: List[str], period: str = "1y",
                            pattern_types: Optional[List[str]] = None) -> Dict[str, Dict[str, List[Pattern]]]:
        """
        Detect patterns for multiple stocks with parallel processing.
        
        Args:
            symbols: List of stock symbols to analyze
            period: Time period for data
            pattern_types: List of pattern types to detect (None for all)
            
        Returns:
            Dictionary mapping symbols to pattern detection results
        """
        logger.info(f"Starting batch pattern detection for {len(symbols)} symbols")
        start_time = datetime.now()
        
        results = {}
        
        if self.config.enable_parallel_processing:
            # Parallel processing
            future_to_symbol = {
                self.executor.submit(self.detect_patterns_single_stock, symbol, period, pattern_types): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol_results = future.result()
                    if symbol_results:
                        results[symbol] = symbol_results
                except Exception as e:
                    logger.error(f"Batch detection failed for {symbol}: {e}")
        else:
            # Sequential processing
            for symbol in symbols:
                try:
                    symbol_results = self.detect_patterns_single_stock(symbol, period, pattern_types)
                    if symbol_results:
                        results[symbol] = symbol_results
                except Exception as e:
                    logger.error(f"Sequential detection failed for {symbol}: {e}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        total_patterns = sum(len(patterns) for symbol_results in results.values() 
                           for patterns in symbol_results.values())
        
        logger.info(f"Batch detection completed: {len(results)} symbols processed, "
                   f"{total_patterns} patterns found in {total_time:.2f}s")
        
        return results
    
    def _validate_patterns(self, patterns: List[Pattern], market_data: MarketData) -> List[Pattern]:
        """
        Apply hybrid validation to detected patterns.
        
        Args:
            patterns: List of patterns to validate
            market_data: Market data for validation
            
        Returns:
            List of validated patterns
        """
        validated_patterns = []
        validation_start = datetime.now()
        
        try:
            for pattern in patterns:
                is_valid, confidence, validation_details = self.hybrid_validator.validate_pattern_hybrid(
                    pattern, market_data
                )
                
                if is_valid:
                    # Update pattern with validation results
                    pattern.confidence = confidence
                    pattern.combined_score = validation_details.get("hybrid_score", pattern.combined_score)
                    validated_patterns.append(pattern)
                    
                    logger.debug(f"Pattern validated: {pattern.type} for {pattern.symbol} "
                               f"(confidence: {confidence:.3f})")
                else:
                    logger.debug(f"Pattern rejected: {pattern.type} for {pattern.symbol} "
                               f"(reason: {validation_details.get('decision_rationale', 'Unknown')})")
        
        except Exception as e:
            logger.error(f"Pattern validation failed: {e}")
            # Return original patterns if validation fails
            return patterns
        
        validation_time = (datetime.now() - validation_start).total_seconds()
        self.stats["validation_times"].append(validation_time)
        
        logger.debug(f"Validated {len(validated_patterns)}/{len(patterns)} patterns "
                    f"in {validation_time:.2f}s")
        
        return validated_patterns
    
    def _organize_validated_patterns(self, patterns: List[Pattern]) -> Dict[str, List[Pattern]]:
        """
        Organize validated patterns by type.
        
        Args:
            patterns: List of validated patterns
            
        Returns:
            Dictionary mapping pattern types to pattern lists
        """
        organized = {}
        
        for pattern in patterns:
            if pattern.type not in organized:
                organized[pattern.type] = []
            organized[pattern.type].append(pattern)
        
        # Sort patterns within each type by combined score (descending)
        for pattern_type in organized:
            organized[pattern_type].sort(key=lambda p: p.combined_score, reverse=True)
        
        return organized
    
    def rank_patterns(self, patterns: Dict[str, List[Pattern]], 
                     ranking_criteria: str = "combined_score") -> List[Pattern]:
        """
        Rank all patterns across types using specified criteria.
        
        Args:
            patterns: Dictionary of patterns by type
            ranking_criteria: Criteria for ranking ("combined_score", "confidence", "risk_reward_ratio")
            
        Returns:
            List of patterns ranked by criteria
        """
        all_patterns = []
        for pattern_list in patterns.values():
            all_patterns.extend(pattern_list)
        
        # Define ranking functions
        ranking_functions = {
            "combined_score": lambda p: p.combined_score,
            "confidence": lambda p: p.confidence,
            "risk_reward_ratio": lambda p: p.risk_reward_ratio,
            "potential_profit": lambda p: p.potential_profit_percent
        }
        
        if ranking_criteria not in ranking_functions:
            logger.warning(f"Unknown ranking criteria: {ranking_criteria}, using combined_score")
            ranking_criteria = "combined_score"
        
        # Sort patterns by selected criteria (descending)
        ranked_patterns = sorted(all_patterns, 
                               key=ranking_functions[ranking_criteria], 
                               reverse=True)
        
        logger.debug(f"Ranked {len(ranked_patterns)} patterns by {ranking_criteria}")
        return ranked_patterns
    
    def filter_patterns(self, patterns: Dict[str, List[Pattern]], 
                       filters: Dict[str, Union[float, bool, str]]) -> Dict[str, List[Pattern]]:
        """
        Filter patterns based on specified criteria.
        
        Args:
            patterns: Dictionary of patterns by type
            filters: Dictionary of filter criteria
            
        Returns:
            Filtered patterns dictionary
        """
        filtered_patterns = {}
        
        for pattern_type, pattern_list in patterns.items():
            filtered_list = []
            
            for pattern in pattern_list:
                # Apply filters
                passes_filters = True
                
                # Confidence filter
                if "min_confidence" in filters:
                    if pattern.confidence < filters["min_confidence"]:
                        passes_filters = False
                
                # Combined score filter
                if "min_combined_score" in filters:
                    if pattern.combined_score < filters["min_combined_score"]:
                        passes_filters = False
                
                # Risk-reward ratio filter
                if "min_risk_reward" in filters:
                    if pattern.risk_reward_ratio < filters["min_risk_reward"]:
                        passes_filters = False
                
                # Volume confirmation filter
                if "require_volume_confirmation" in filters:
                    if filters["require_volume_confirmation"] and not pattern.volume_confirmation:
                        passes_filters = False
                
                # Pattern status filter
                if "status" in filters:
                    if pattern.status != filters["status"]:
                        passes_filters = False
                
                # Bullish/bearish filter
                if "pattern_direction" in filters:
                    direction = filters["pattern_direction"].lower()
                    if direction == "bullish" and not pattern.is_bullish:
                        passes_filters = False
                    elif direction == "bearish" and not pattern.is_bearish:
                        passes_filters = False
                
                if passes_filters:
                    filtered_list.append(pattern)
            
            if filtered_list:
                filtered_patterns[pattern_type] = filtered_list
        
        total_filtered = sum(len(patterns) for patterns in filtered_patterns.values())
        total_original = sum(len(patterns) for patterns in patterns.values())
        
        logger.debug(f"Filtered patterns: {total_filtered}/{total_original} patterns passed filters")
        return filtered_patterns
    
    def get_detection_summary(self, patterns: Dict[str, List[Pattern]]) -> Dict:
        """
        Generate summary statistics for detection results.
        
        Args:
            patterns: Dictionary of patterns by type
            
        Returns:
            Summary statistics dictionary
        """
        if not patterns:
            return {"total_patterns": 0}
        
        all_patterns = []
        for pattern_list in patterns.values():
            all_patterns.extend(pattern_list)
        
        # Calculate statistics
        confidences = [p.confidence for p in all_patterns]
        combined_scores = [p.combined_score for p in all_patterns]
        risk_rewards = [p.risk_reward_ratio for p in all_patterns if p.risk_reward_ratio > 0]
        
        summary = {
            "total_patterns": len(all_patterns),
            "patterns_by_type": {ptype: len(plist) for ptype, plist in patterns.items()},
            "confidence_stats": {
                "mean": np.mean(confidences) if confidences else 0,
                "median": np.median(confidences) if confidences else 0,
                "min": np.min(confidences) if confidences else 0,
                "max": np.max(confidences) if confidences else 0
            },
            "combined_score_stats": {
                "mean": np.mean(combined_scores) if combined_scores else 0,
                "median": np.median(combined_scores) if combined_scores else 0,
                "min": np.min(combined_scores) if combined_scores else 0,
                "max": np.max(combined_scores) if combined_scores else 0
            },
            "risk_reward_stats": {
                "mean": np.mean(risk_rewards) if risk_rewards else 0,
                "median": np.median(risk_rewards) if risk_rewards else 0,
                "min": np.min(risk_rewards) if risk_rewards else 0,
                "max": np.max(risk_rewards) if risk_rewards else 0
            },
            "volume_confirmation_rate": sum(1 for p in all_patterns if p.volume_confirmation) / len(all_patterns) if all_patterns else 0,
            "bullish_patterns": sum(1 for p in all_patterns if p.is_bullish),
            "bearish_patterns": sum(1 for p in all_patterns if p.is_bearish)
        }
        
        return summary
    
    def get_engine_stats(self) -> Dict:
        """
        Get engine performance statistics.
        
        Returns:
            Engine statistics dictionary
        """
        stats = self.stats.copy()
        
        if stats["detection_times"]:
            stats["avg_detection_time"] = np.mean(stats["detection_times"])
            stats["total_detection_time"] = sum(stats["detection_times"])
        
        if stats["validation_times"]:
            stats["avg_validation_time"] = np.mean(stats["validation_times"])
            stats["total_validation_time"] = sum(stats["validation_times"])
        
        if stats["total_scans"] > 0:
            stats["success_rate"] = stats["successful_scans"] / stats["total_scans"]
        
        if stats["patterns_detected"] > 0:
            stats["validation_rate"] = stats["patterns_validated"] / stats["patterns_detected"]
        
        # Add cache statistics
        stats["cache_stats"] = self.data_manager.get_cache_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear data cache."""
        self.data_manager.clear_cache()
        logger.info("Data cache cleared")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)