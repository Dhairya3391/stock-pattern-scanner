"""
Hybrid Validation System for Advanced Pattern Scanner.

This module provides a hybrid validation system that combines traditional
rule-based pattern detection with ML-based validation to create a robust
pattern detection pipeline with reduced false positives.
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from datetime import datetime

from ..core.models import Pattern, PatternConfig, MarketData
from .model_manager import ModelManager
from .pattern_validator import PatternValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridValidator:
    """
    Hybrid validation system combining traditional and ML approaches.
    
    This class orchestrates the validation process by first using traditional
    rule-based algorithms for pattern detection, then applying ML models for
    validation and confidence scoring.
    """
    
    def __init__(self, config: PatternConfig):
        """
        Initialize the hybrid validator.
        
        Args:
            config: Pattern detection configuration
        """
        self.config = config
        
        # Initialize ML components
        self.model_manager = ModelManager(config)
        self.pattern_validator = PatternValidator(config, self.model_manager)
        
        # Load models
        self.models_loaded = self.model_manager.load_models()
        
        # Validation strategy weights
        self.strategy_weights = {
            "traditional_primary": 0.4,    # Traditional algorithm confidence
            "ml_validation": 0.35,         # ML model validation
            "technical_confirmation": 0.15, # Technical indicators
            "volume_confirmation": 0.10     # Volume analysis
        }
        
        # Pattern-specific thresholds
        self.pattern_thresholds = {
            "Head and Shoulders": {
                "min_traditional": 0.70,
                "min_ml": 0.75,
                "min_combined": 0.72
            },
            "Double Bottom": {
                "min_traditional": 0.65,
                "min_ml": 0.70,
                "min_combined": 0.68
            },
            "Double Top": {
                "min_traditional": 0.65,
                "min_ml": 0.70,
                "min_combined": 0.68
            },
            "Cup and Handle": {
                "min_traditional": 0.75,
                "min_ml": 0.80,
                "min_combined": 0.77
            },
            "Ascending Triangle": {
                "min_traditional": 0.60,
                "min_ml": 0.65,
                "min_combined": 0.62
            },
            "Descending Triangle": {
                "min_traditional": 0.60,
                "min_ml": 0.65,
                "min_combined": 0.62
            }
        }
        
        logger.info(f"HybridValidator initialized. Models loaded: {self.models_loaded}")
    
    def validate_pattern_hybrid(self, pattern: Pattern, 
                              market_data: MarketData) -> Tuple[bool, float, Dict]:
        """
        Perform hybrid validation on a detected pattern.
        
        This is the main validation method that combines traditional rule-based
        validation with ML-based confirmation.
        
        Args:
            pattern: Pattern detected by traditional algorithms
            market_data: Market data for the pattern
            
        Returns:
            Tuple of (is_valid, confidence_score, validation_details)
        """
        validation_details = {
            "traditional_validation": {},
            "ml_validation": {},
            "hybrid_score": 0.0,
            "validation_components": {},
            "decision_rationale": "",
            "timestamp": datetime.now()
        }
        
        try:
            # Step 1: Traditional validation (already done, use existing score)
            traditional_score = pattern.traditional_score
            validation_details["traditional_validation"] = {
                "score": traditional_score,
                "passed": traditional_score >= self._get_threshold(pattern.type, "min_traditional")
            }
            
            # Step 2: ML-based validation
            if self.models_loaded:
                ml_valid, ml_score, ml_details = self.pattern_validator.validate_pattern(
                    pattern, market_data
                )
                validation_details["ml_validation"] = {
                    "valid": ml_valid,
                    "score": ml_score,
                    "details": ml_details
                }
            else:
                # Fallback if no models loaded
                ml_score = 0.5  # Neutral score
                validation_details["ml_validation"] = {
                    "valid": True,
                    "score": ml_score,
                    "details": {"note": "No ML models loaded, using neutral score"}
                }
            
            # Step 3: Calculate hybrid score
            hybrid_score = self._calculate_hybrid_score(
                pattern, traditional_score, ml_score, validation_details
            )
            
            validation_details["hybrid_score"] = hybrid_score
            
            # Step 4: Make final decision
            is_valid, decision_rationale = self._make_hybrid_decision(
                pattern, traditional_score, ml_score, hybrid_score
            )
            
            validation_details["decision_rationale"] = decision_rationale
            
            # Update pattern with hybrid results
            pattern.combined_score = hybrid_score
            pattern.confidence = ml_score if self.models_loaded else traditional_score
            
            logger.debug(f"Hybrid validation: {pattern.type} - "
                        f"Traditional: {traditional_score:.3f}, ML: {ml_score:.3f}, "
                        f"Hybrid: {hybrid_score:.3f}, Valid: {is_valid}")
            
            return is_valid, hybrid_score, validation_details
            
        except Exception as e:
            logger.error(f"Hybrid validation failed: {e}")
            validation_details["error"] = str(e)
            return False, 0.0, validation_details
    
    def _calculate_hybrid_score(self, pattern: Pattern, traditional_score: float, 
                              ml_score: float, validation_details: Dict) -> float:
        """
        Calculate the hybrid validation score.
        
        Args:
            pattern: Pattern being validated
            traditional_score: Score from traditional algorithm
            ml_score: Score from ML validation
            validation_details: Validation details dictionary
            
        Returns:
            Hybrid validation score (0-1)
        """
        # Base hybrid score using weighted combination
        base_score = (
            traditional_score * self.strategy_weights["traditional_primary"] +
            ml_score * self.strategy_weights["ml_validation"]
        )
        
        # Add technical confirmation component
        tech_score = 0.5  # Default neutral
        if "ml_validation" in validation_details and "details" in validation_details["ml_validation"]:
            ml_details = validation_details["ml_validation"]["details"]
            if "technical_validation" in ml_details:
                tech_score = ml_details["technical_validation"].get("score", 0.5)
        
        # Add volume confirmation component
        volume_score = 0.5  # Default neutral
        if "ml_validation" in validation_details and "details" in validation_details["ml_validation"]:
            ml_details = validation_details["ml_validation"]["details"]
            if "volume_validation" in ml_details:
                volume_score = ml_details["volume_validation"].get("score", 0.5)
        
        # Calculate final hybrid score
        hybrid_score = (
            base_score +
            tech_score * self.strategy_weights["technical_confirmation"] +
            volume_score * self.strategy_weights["volume_confirmation"]
        )
        
        # Apply pattern-specific adjustments
        hybrid_score = self._apply_pattern_adjustments(pattern, hybrid_score)
        
        # Ensure score is in valid range
        hybrid_score = max(0.0, min(1.0, hybrid_score))
        
        # Store component scores
        validation_details["validation_components"] = {
            "traditional_component": traditional_score * self.strategy_weights["traditional_primary"],
            "ml_component": ml_score * self.strategy_weights["ml_validation"],
            "technical_component": tech_score * self.strategy_weights["technical_confirmation"],
            "volume_component": volume_score * self.strategy_weights["volume_confirmation"],
            "pattern_adjustment": hybrid_score - base_score - 
                                tech_score * self.strategy_weights["technical_confirmation"] - 
                                volume_score * self.strategy_weights["volume_confirmation"]
        }
        
        return hybrid_score
    
    def _apply_pattern_adjustments(self, pattern: Pattern, base_score: float) -> float:
        """
        Apply pattern-specific adjustments to the hybrid score.
        
        Args:
            pattern: Pattern being validated
            base_score: Base hybrid score
            
        Returns:
            Adjusted hybrid score
        """
        adjusted_score = base_score
        
        # Pattern-specific adjustments
        if pattern.type == "Head and Shoulders":
            # H&S patterns require strong volume confirmation
            if pattern.volume_confirmation:
                adjusted_score += 0.05
            else:
                adjusted_score -= 0.10
                
        elif pattern.type in ["Double Bottom", "Double Top"]:
            # Double patterns benefit from symmetry
            if len(pattern.key_points) >= 2:
                # Check symmetry (simplified)
                if pattern.pattern_height > 0:
                    adjusted_score += 0.03
                    
        elif pattern.type == "Cup and Handle":
            # Cup and handle requires proper depth and duration
            if (pattern.duration_days >= 30 and 
                pattern.pattern_height >= pattern.entry_price * 0.15):
                adjusted_score += 0.05
            else:
                adjusted_score -= 0.05
                
        elif "Triangle" in pattern.type:
            # Triangles need convergence and breakout volume
            if pattern.volume_confirmation:
                adjusted_score += 0.08
            else:
                adjusted_score -= 0.05
        
        # Duration-based adjustments
        if pattern.duration_days < self.config.min_pattern_duration:
            adjusted_score -= 0.15  # Penalize too-short patterns
        elif pattern.duration_days > self.config.max_pattern_duration:
            adjusted_score -= 0.10  # Penalize too-long patterns
        
        # Risk-reward ratio adjustment
        if pattern.risk_reward_ratio >= 2.0:
            adjusted_score += 0.05
        elif pattern.risk_reward_ratio < 1.0:
            adjusted_score -= 0.10
        
        return adjusted_score
    
    def _make_hybrid_decision(self, pattern: Pattern, traditional_score: float, 
                            ml_score: float, hybrid_score: float) -> Tuple[bool, str]:
        """
        Make the final validation decision based on all scores.
        
        Args:
            pattern: Pattern being validated
            traditional_score: Traditional algorithm score
            ml_score: ML validation score
            hybrid_score: Hybrid validation score
            
        Returns:
            Tuple of (is_valid, decision_rationale)
        """
        pattern_thresholds = self.pattern_thresholds.get(pattern.type, {
            "min_traditional": self.config.min_traditional_score,
            "min_ml": self.config.min_confidence,
            "min_combined": self.config.min_combined_score
        })
        
        # Check individual thresholds
        traditional_pass = traditional_score >= pattern_thresholds["min_traditional"]
        ml_pass = ml_score >= pattern_thresholds["min_ml"]
        hybrid_pass = hybrid_score >= pattern_thresholds["min_combined"]
        
        # Decision logic
        if hybrid_pass and traditional_pass:
            if ml_pass:
                return True, "All validation criteria passed"
            else:
                return True, "Traditional and hybrid validation passed (ML threshold not met but acceptable)"
        
        elif hybrid_pass and ml_pass:
            return True, "ML and hybrid validation passed (traditional threshold not met but acceptable)"
        
        elif traditional_pass and ml_pass:
            # Both individual methods pass but hybrid doesn't - investigate
            if hybrid_score >= (pattern_thresholds["min_combined"] - 0.05):
                return True, "Individual validations passed, hybrid score close to threshold"
            else:
                return False, "Individual validations passed but hybrid score too low"
        
        elif hybrid_pass:
            # Only hybrid passes - be conservative
            if hybrid_score >= (pattern_thresholds["min_combined"] + 0.05):
                return True, "Hybrid validation passed with high confidence"
            else:
                return False, "Only hybrid validation passed, insufficient confidence"
        
        else:
            # Determine primary reason for failure
            if not traditional_pass and not ml_pass:
                return False, "Both traditional and ML validation failed"
            elif not traditional_pass:
                return False, "Traditional validation failed"
            elif not ml_pass:
                return False, "ML validation failed"
            else:
                return False, "Hybrid score below threshold"
    
    def _get_threshold(self, pattern_type: str, threshold_type: str) -> float:
        """Get threshold value for pattern type and threshold type."""
        pattern_thresholds = self.pattern_thresholds.get(pattern_type, {})
        
        if threshold_type in pattern_thresholds:
            return pattern_thresholds[threshold_type]
        
        # Fallback to config defaults
        if threshold_type == "min_traditional":
            return self.config.min_traditional_score
        elif threshold_type == "min_ml":
            return self.config.min_confidence
        elif threshold_type == "min_combined":
            return self.config.min_combined_score
        else:
            return 0.7  # Default threshold
    
    def batch_validate_hybrid(self, patterns: List[Pattern], 
                            market_data_list: List[MarketData]) -> List[Tuple[bool, float, Dict]]:
        """
        Perform hybrid validation on multiple patterns.
        
        Args:
            patterns: List of patterns to validate
            market_data_list: List of corresponding market data
            
        Returns:
            List of validation results
        """
        results = []
        
        for i, (pattern, market_data) in enumerate(zip(patterns, market_data_list)):
            try:
                result = self.validate_pattern_hybrid(pattern, market_data)
                results.append(result)
                
                if i % 10 == 0 and i > 0:
                    logger.debug(f"Processed {i} patterns in batch validation")
                    
            except Exception as e:
                logger.warning(f"Batch hybrid validation failed for pattern {i}: {e}")
                results.append((False, 0.0, {"error": str(e)}))
        
        return results
    
    def get_validation_statistics(self, validation_results: List[Tuple[bool, float, Dict]]) -> Dict:
        """
        Generate comprehensive statistics for validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Statistics dictionary
        """
        if not validation_results:
            return {}
        
        valid_results = [r for r in validation_results if r[0]]
        invalid_results = [r for r in validation_results if not r[0]]
        
        all_scores = [r[1] for r in validation_results]
        valid_scores = [r[1] for r in valid_results]
        invalid_scores = [r[1] for r in invalid_results]
        
        # Basic statistics
        stats = {
            "total_patterns": len(validation_results),
            "valid_patterns": len(valid_results),
            "invalid_patterns": len(invalid_results),
            "validation_rate": len(valid_results) / len(validation_results) if validation_results else 0.0,
            
            "score_statistics": {
                "all_patterns": {
                    "mean": np.mean(all_scores) if all_scores else 0.0,
                    "median": np.median(all_scores) if all_scores else 0.0,
                    "std": np.std(all_scores) if all_scores else 0.0,
                    "min": np.min(all_scores) if all_scores else 0.0,
                    "max": np.max(all_scores) if all_scores else 0.0
                },
                "valid_patterns": {
                    "mean": np.mean(valid_scores) if valid_scores else 0.0,
                    "median": np.median(valid_scores) if valid_scores else 0.0,
                    "std": np.std(valid_scores) if valid_scores else 0.0,
                    "min": np.min(valid_scores) if valid_scores else 0.0,
                    "max": np.max(valid_scores) if valid_scores else 0.0
                },
                "invalid_patterns": {
                    "mean": np.mean(invalid_scores) if invalid_scores else 0.0,
                    "median": np.median(invalid_scores) if invalid_scores else 0.0,
                    "std": np.std(invalid_scores) if invalid_scores else 0.0,
                    "min": np.min(invalid_scores) if invalid_scores else 0.0,
                    "max": np.max(invalid_scores) if invalid_scores else 0.0
                }
            }
        }
        
        # Component analysis
        component_stats = {
            "traditional_scores": [],
            "ml_scores": [],
            "technical_scores": [],
            "volume_scores": []
        }
        
        for _, _, details in validation_results:
            if "validation_components" in details:
                components = details["validation_components"]
                component_stats["traditional_scores"].append(
                    components.get("traditional_component", 0.0)
                )
                component_stats["ml_scores"].append(
                    components.get("ml_component", 0.0)
                )
                component_stats["technical_scores"].append(
                    components.get("technical_component", 0.0)
                )
                component_stats["volume_scores"].append(
                    components.get("volume_component", 0.0)
                )
        
        # Add component statistics
        for component, scores in component_stats.items():
            if scores:
                stats[f"{component}_stats"] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "contribution": np.mean(scores) / np.mean(all_scores) if np.mean(all_scores) > 0 else 0.0
                }
        
        return stats
    
    def update_thresholds(self, pattern_type: str, thresholds: Dict[str, float]):
        """
        Update validation thresholds for a specific pattern type.
        
        Args:
            pattern_type: Type of pattern to update
            thresholds: Dictionary of threshold values
        """
        if pattern_type not in self.pattern_thresholds:
            self.pattern_thresholds[pattern_type] = {}
        
        self.pattern_thresholds[pattern_type].update(thresholds)
        logger.info(f"Updated thresholds for {pattern_type}: {thresholds}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the hybrid validation system.
        
        Returns:
            System information dictionary
        """
        info = {
            "models_loaded": self.models_loaded,
            "strategy_weights": self.strategy_weights,
            "pattern_thresholds": self.pattern_thresholds,
            "config": {
                "min_confidence": self.config.min_confidence,
                "min_traditional_score": self.config.min_traditional_score,
                "min_combined_score": self.config.min_combined_score
            }
        }
        
        if self.models_loaded:
            info["model_details"] = self.model_manager.get_model_info()
        
        return info