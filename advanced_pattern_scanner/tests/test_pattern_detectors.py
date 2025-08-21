"""
Test suite for pattern detection algorithms.

Tests each pattern detector using exact examples from reference documents
to validate implementation accuracy.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
import logging

from ..core.models import MarketData, PatternConfig
from ..patterns.head_shoulders import HeadShouldersDetector
from ..patterns.double_bottom import DoubleBottomDetector
from ..patterns.cup_handle import CupHandleDetector

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestPatternDetectors(unittest.TestCase):
    """Test suite for all pattern detection algorithms."""
    
    def setUp(self):
        """Set up test configuration and detectors."""
        self.config = PatternConfig()
        self.hs_detector = HeadShouldersDetector(self.config)
        self.db_detector = DoubleBottomDetector(self.config)
        self.ch_detector = CupHandleDetector(self.config)
    
    def create_market_data(self, symbol: str, prices: np.ndarray, volumes: np.ndarray = None) -> MarketData:
        """Helper to create MarketData object for testing."""
        if volumes is None:
            volumes = np.ones(len(prices)) * 1000  # Default volume
        
        # Create OHLCV data (using close price for all OHLC for simplicity)
        ohlcv = np.column_stack([prices, prices, prices, prices, volumes])
        
        # Create timestamps
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(len(prices))]
        
        return MarketData(
            symbol=symbol,
            timeframe="1d",
            data=ohlcv,
            timestamps=timestamps
        )
    
    def test_head_shoulders_basic_functionality(self):
        """Test basic Head & Shoulders detection functionality."""
        logger.info("Testing Head & Shoulders basic functionality")
        
        # Create a synthetic H&S pattern
        # Left shoulder: 100, Head: 110, Right shoulder: 102, then decline
        prices = np.array([
            90, 95, 100, 95, 90,      # Left shoulder formation
            95, 105, 110, 105, 95,    # Head formation  
            100, 102, 100, 95,        # Right shoulder formation
            90, 85, 80, 75            # Neckline break and decline
        ])
        
        volumes = np.array([
            1000, 1200, 1500, 1200, 1000,  # Normal volume
            1200, 1800, 2000, 1800, 1200,  # High volume at head
            1100, 1300, 1100, 1000,        # Lower volume at right shoulder
            1600, 1800, 2200, 2500         # High volume on break
        ])
        
        market_data = self.create_market_data("TEST_HS", prices, volumes)
        patterns = self.hs_detector.detect_pattern(market_data)
        
        self.assertGreater(len(patterns), 0, "Should detect at least one H&S pattern")
        
        if patterns:
            pattern = patterns[0]
            self.assertEqual(pattern.type, "Head and Shoulders")
            self.assertEqual(pattern.status, "confirmed")
            self.assertGreater(pattern.confidence, 0.5)
            logger.info(f"H&S pattern detected with confidence: {pattern.confidence}")
    
    def test_double_bottom_stock_abc_example(self):
        """Test Double Bottom with exact Stock ABC example from reference."""
        logger.info("Testing Double Bottom with Stock ABC example")
        
        # Exact prices from doubleBottom.md Stock ABC example
        abc_prices = np.array([110, 105, 100, 105, 110, 115, 120, 115, 110, 105, 102, 105, 110, 115, 125])
        
        market_data = self.create_market_data("ABC", abc_prices)
        patterns = self.db_detector.detect_pattern(market_data)
        
        self.assertGreater(len(patterns), 0, "Should detect double bottom in Stock ABC example")
        
        if patterns:
            pattern = patterns[0]
            self.assertEqual(pattern.type, "Double Bottom")
            self.assertEqual(pattern.status, "confirmed")
            
            # Verify key points match expected values
            key_points = pattern.key_points
            first_bottom_price = key_points[0][1]
            second_bottom_price = key_points[2][1]
            
            # Should detect bottoms around 100 and 102 (within tolerance)
            self.assertAlmostEqual(first_bottom_price, 100, delta=2)
            self.assertAlmostEqual(second_bottom_price, 102, delta=2)
            
            logger.info(f"Stock ABC: Detected bottoms at {first_bottom_price} and {second_bottom_price}")
    
    def test_double_bottom_stock_xyz_example(self):
        """Test Double Bottom with exact Stock XYZ example (should not confirm)."""
        logger.info("Testing Double Bottom with Stock XYZ example")
        
        # Exact prices from doubleBottom.md Stock XYZ example
        xyz_prices = np.array([100, 98, 95, 97, 99, 101, 103, 100, 98, 96, 94, 96, 98, 100, 102])
        
        market_data = self.create_market_data("XYZ", xyz_prices)
        patterns = self.db_detector.detect_pattern(market_data)
        
        # Should not detect confirmed pattern (no breakout above 103)
        self.assertEqual(len(patterns), 0, "Should not detect confirmed double bottom in Stock XYZ (no breakout)")
        logger.info("Stock XYZ: Correctly identified no confirmed pattern (no breakout above 103)")
    
    def test_double_bottom_reference_test_method(self):
        """Test the built-in reference test method."""
        logger.info("Testing Double Bottom reference test method")
        
        result = self.db_detector.test_with_reference_examples()
        self.assertTrue(result, "Reference examples should pass validation")
    
    def test_cup_handle_basic_functionality(self):
        """Test basic Cup & Handle detection functionality."""
        logger.info("Testing Cup & Handle basic functionality")
        
        # Create a synthetic Cup & Handle pattern
        # Uptrend, then cup formation, then handle, then breakout
        prices = np.array([
            # Initial uptrend
            50, 55, 60, 65, 70, 75, 80,
            # Cup formation (U-shaped decline and recovery) - shallower cup
            79, 77, 75, 73, 71, 70, 71, 73, 75, 77, 79,
            # Handle formation (small consolidation)
            78, 77, 76, 77, 78,
            # Breakout
            82, 85, 88, 90
        ])
        
        volumes = np.array([
            # Normal volume during uptrend
            1000, 1100, 1200, 1300, 1400, 1500, 1600,
            # Decreasing volume during cup
            1400, 1200, 1000, 800, 600, 500, 600, 800, 1000, 1200, 1400,
            # Low volume during handle
            900, 800, 700, 800, 900,
            # High volume on breakout
            2000, 2200, 2400, 2600
        ])
        
        market_data = self.create_market_data("TEST_CH", prices, volumes)
        patterns = self.ch_detector.detect_pattern(market_data)
        
        self.assertGreater(len(patterns), 0, "Should detect at least one Cup & Handle pattern")
        
        if patterns:
            pattern = patterns[0]
            self.assertEqual(pattern.type, "Cup and Handle")
            self.assertEqual(pattern.status, "confirmed")
            self.assertGreater(pattern.confidence, 0.5)
            logger.info(f"Cup & Handle pattern detected with confidence: {pattern.confidence}")
    
    def test_pattern_config_validation(self):
        """Test pattern configuration validation."""
        logger.info("Testing pattern configuration validation")
        
        # Test valid configuration
        valid_config = PatternConfig()
        errors = valid_config.validate()
        self.assertEqual(len(errors), 0, "Valid configuration should have no errors")
        
        # Test invalid configuration
        invalid_config = PatternConfig(
            min_confidence=-0.1,  # Invalid: negative
            max_pattern_duration=10,  # Invalid: less than min
            min_pattern_duration=20
        )
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0, "Invalid configuration should have errors")
    
    def test_pattern_scoring_and_metrics(self):
        """Test pattern scoring and risk metrics calculation."""
        logger.info("Testing pattern scoring and metrics")
        
        # Create a simple double bottom for testing
        prices = np.array([100, 95, 90, 95, 100, 105, 110, 105, 100, 95, 92, 95, 100, 105, 115])
        market_data = self.create_market_data("METRICS_TEST", prices)
        
        patterns = self.db_detector.detect_pattern(market_data)
        
        if patterns:
            pattern = patterns[0]
            
            # Test that all required metrics are calculated
            self.assertIsNotNone(pattern.confidence)
            self.assertIsNotNone(pattern.traditional_score)
            self.assertIsNotNone(pattern.combined_score)
            self.assertIsNotNone(pattern.risk_reward_ratio)
            self.assertIsNotNone(pattern.pattern_height)
            self.assertIsNotNone(pattern.duration_days)
            
            # Test that scores are within valid ranges
            self.assertGreaterEqual(pattern.confidence, 0)
            self.assertLessEqual(pattern.confidence, 1)
            self.assertGreaterEqual(pattern.traditional_score, 0)
            self.assertLessEqual(pattern.traditional_score, 1)
            
            # Test bullish/bearish classification
            self.assertTrue(pattern.is_bullish)
            self.assertFalse(pattern.is_bearish)
            
            logger.info(f"Pattern metrics: confidence={pattern.confidence:.2f}, "
                       f"risk_reward={pattern.risk_reward_ratio:.2f}, "
                       f"duration={pattern.duration_days} days")
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        logger.info("Testing insufficient data handling")
        
        # Test with very short price series
        short_prices = np.array([100, 105, 110])
        short_data = self.create_market_data("SHORT", short_prices)
        
        hs_patterns = self.hs_detector.detect_pattern(short_data)
        db_patterns = self.db_detector.detect_pattern(short_data)
        ch_patterns = self.ch_detector.detect_pattern(short_data)
        
        # All should handle insufficient data gracefully
        self.assertEqual(len(hs_patterns), 0, "H&S should handle insufficient data")
        self.assertEqual(len(db_patterns), 0, "Double Bottom should handle insufficient data")
        self.assertEqual(len(ch_patterns), 0, "Cup & Handle should handle insufficient data")
    
    def test_edge_cases(self):
        """Test various edge cases and error conditions."""
        logger.info("Testing edge cases")
        
        # Test with flat prices (no patterns)
        flat_prices = np.array([100] * 50)
        flat_data = self.create_market_data("FLAT", flat_prices)
        
        patterns = self.db_detector.detect_pattern(flat_data)
        self.assertEqual(len(patterns), 0, "Should not detect patterns in flat prices")
        
        # Test with extreme volatility
        volatile_prices = np.random.normal(100, 20, 100)
        volatile_prices = np.abs(volatile_prices)  # Ensure positive prices
        volatile_data = self.create_market_data("VOLATILE", volatile_prices)
        
        # Should not crash, may or may not find patterns
        try:
            patterns = self.hs_detector.detect_pattern(volatile_data)
            logger.info(f"Volatile data test: Found {len(patterns)} H&S patterns")
        except Exception as e:
            self.fail(f"Should handle volatile data gracefully: {e}")


def run_pattern_tests():
    """Run all pattern detection tests."""
    logger.info("Starting pattern detection tests")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatternDetectors)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    if result.wasSuccessful():
        logger.info("All pattern detection tests passed!")
        return True
    else:
        logger.error(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    run_pattern_tests()