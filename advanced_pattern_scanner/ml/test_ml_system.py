"""
Test script for ML validation system.

This script tests the ML validation system components to ensure they work
correctly with the existing pattern detection infrastructure.
"""

import sys
import logging
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from advanced_pattern_scanner.core.models import Pattern, PatternConfig, MarketData
from advanced_pattern_scanner.ml import (
    ModelManager, PatternValidator, HybridValidator, 
    SyntheticDataGenerator, ModelTrainer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_pattern() -> Pattern:
    """Create a test pattern for validation."""
    return Pattern(
        type="Head and Shoulders",
        symbol="TEST",
        timeframe="1d",
        key_points=[(10, 100.0), (20, 110.0), (30, 105.0), (40, 115.0), (50, 105.0), (60, 108.0), (70, 95.0)],
        confidence=0.0,  # Will be set by ML
        traditional_score=0.75,
        combined_score=0.0,  # Will be calculated
        entry_price=95.0,
        target_price=85.0,
        stop_loss=100.0,
        risk_reward_ratio=1.5,
        formation_start=datetime.now() - timedelta(days=70),
        formation_end=datetime.now() - timedelta(days=10),
        status="confirmed",
        volume_confirmation=True,
        avg_volume_ratio=1.3,
        pattern_height=20.0,
        duration_days=60,
        detection_method="traditional"
    )


def create_test_market_data() -> MarketData:
    """Create test market data."""
    # Generate synthetic OHLCV data
    np.random.seed(42)
    length = 100
    base_price = 100.0
    
    # Generate price series
    returns = np.random.normal(0, 0.02, length)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV array
    ohlcv = np.zeros((length, 5))
    for i in range(length):
        close = prices[i]
        open_price = prices[i-1] if i > 0 else close
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.lognormal(13, 0.3)  # Log-normal volume
        
        ohlcv[i] = [open_price, high, low, close, volume]
    
    # Create timestamps
    timestamps = [datetime.now() - timedelta(days=length-i) for i in range(length)]
    
    return MarketData(
        symbol="TEST",
        timeframe="1d",
        data=ohlcv,
        timestamps=timestamps
    )


def test_model_manager():
    """Test the ModelManager component."""
    logger.info("Testing ModelManager...")
    
    config = PatternConfig()
    model_manager = ModelManager(config)
    
    # Test model loading (will create synthetic models)
    success = model_manager.load_models()
    assert success, "Model loading failed"
    
    # Test prediction
    test_features = np.random.randn(1, 600)  # Flattened features
    pattern_name, confidence = model_manager.predict_pattern(test_features)
    
    assert isinstance(pattern_name, str), "Pattern name should be string"
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    
    logger.info(f"ModelManager test passed. Predicted: {pattern_name}, Confidence: {confidence:.3f}")


def test_pattern_validator():
    """Test the PatternValidator component."""
    logger.info("Testing PatternValidator...")
    
    config = PatternConfig()
    model_manager = ModelManager(config)
    model_manager.load_models()
    
    validator = PatternValidator(config, model_manager)
    
    # Create test data
    pattern = create_test_pattern()
    market_data = create_test_market_data()
    
    # Test validation
    is_valid, confidence, details = validator.validate_pattern(pattern, market_data)
    
    assert isinstance(is_valid, bool), "Validation result should be boolean"
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    assert isinstance(details, dict), "Details should be dictionary"
    
    logger.info(f"PatternValidator test passed. Valid: {is_valid}, Confidence: {confidence:.3f}")


def test_hybrid_validator():
    """Test the HybridValidator component."""
    logger.info("Testing HybridValidator...")
    
    config = PatternConfig()
    hybrid_validator = HybridValidator(config)
    
    # Create test data
    pattern = create_test_pattern()
    market_data = create_test_market_data()
    
    # Test hybrid validation
    is_valid, confidence, details = hybrid_validator.validate_pattern_hybrid(pattern, market_data)
    
    assert isinstance(is_valid, bool), "Validation result should be boolean"
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    assert isinstance(details, dict), "Details should be dictionary"
    assert "hybrid_score" in details, "Details should contain hybrid score"
    
    logger.info(f"HybridValidator test passed. Valid: {is_valid}, Confidence: {confidence:.3f}")


def test_synthetic_data_generator():
    """Test the SyntheticDataGenerator component."""
    logger.info("Testing SyntheticDataGenerator...")
    
    generator = SyntheticDataGenerator()
    
    # Test dataset generation
    features, labels, pattern_names = generator.generate_dataset(num_samples=96, sequence_length=60)  # 96 = 6 patterns * 16 samples each
    
    assert features.shape[0] == 96, "Should generate 96 samples (6 patterns * 16 each)"
    assert len(labels) == 96, "Should have 96 labels"
    assert len(pattern_names) > 0, "Should have pattern names"
    assert features.shape[1] == 600, "Features should be flattened (60 * 10)"
    
    logger.info(f"SyntheticDataGenerator test passed. Generated {features.shape[0]} samples with {len(pattern_names)} pattern types")


def test_model_trainer():
    """Test the ModelTrainer component (without full training)."""
    logger.info("Testing ModelTrainer initialization...")
    
    config = PatternConfig()
    trainer = ModelTrainer(config)
    
    # Test initialization
    assert trainer.config == config, "Config should be stored"
    assert trainer.device is not None, "Device should be set"
    
    logger.info(f"ModelTrainer test passed. Device: {trainer.device}")


def test_integration():
    """Test integration between components."""
    logger.info("Testing component integration...")
    
    config = PatternConfig()
    
    # Create hybrid validator (integrates all components)
    hybrid_validator = HybridValidator(config)
    
    # Create test data
    patterns = [create_test_pattern() for _ in range(3)]
    market_data_list = [create_test_market_data() for _ in range(3)]
    
    # Test batch validation
    results = hybrid_validator.batch_validate_hybrid(patterns, market_data_list)
    
    assert len(results) == 3, "Should validate all 3 patterns"
    
    for is_valid, confidence, details in results:
        assert isinstance(is_valid, bool), "Each result should have boolean validity"
        assert 0.0 <= confidence <= 1.0, "Each result should have valid confidence"
        assert isinstance(details, dict), "Each result should have details"
    
    # Test statistics
    stats = hybrid_validator.get_validation_statistics(results)
    assert "total_patterns" in stats, "Stats should include total patterns"
    assert stats["total_patterns"] == 3, "Should report 3 total patterns"
    
    logger.info("Integration test passed. All components work together correctly.")


def main():
    """Run all tests."""
    logger.info("Starting ML system tests...")
    
    try:
        test_model_manager()
        test_pattern_validator()
        test_hybrid_validator()
        test_synthetic_data_generator()
        test_model_trainer()
        test_integration()
        
        logger.info("All ML system tests passed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()