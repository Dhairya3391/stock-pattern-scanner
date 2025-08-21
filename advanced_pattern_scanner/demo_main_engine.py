"""
Demo script for the main pattern detection engine and user interface.

This script demonstrates the complete workflow of the pattern detection system
including the main engine, pattern scorer, and basic visualization.
"""

import logging
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

from core.models import PatternConfig
from core.pattern_engine import PatternEngine
from core.pattern_scorer import PatternScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_single_stock_detection():
    """Demonstrate pattern detection for a single stock."""
    print("\n" + "="*60)
    print("DEMO: Single Stock Pattern Detection")
    print("="*60)
    
    # Create configuration
    config = PatternConfig(
        min_confidence=0.6,
        min_combined_score=0.65,
        volume_confirmation_required=True,
        min_volume_ratio=1.2
    )
    
    # Initialize engines
    pattern_engine = PatternEngine(config)
    pattern_scorer = PatternScorer(config)
    
    # Test with Apple stock
    symbol = "AAPL"
    print(f"Analyzing {symbol}...")
    
    try:
        # Detect patterns
        results = pattern_engine.detect_patterns_single_stock(
            symbol=symbol,
            period="1y",
            pattern_types=["Head and Shoulders", "Double Bottom", "Cup and Handle"]
        )
        
        if results:
            print(f"\n✅ Found patterns in {symbol}:")
            
            for pattern_type, patterns in results.items():
                print(f"\n📊 {pattern_type} Patterns ({len(patterns)} found):")
                
                for i, pattern in enumerate(patterns, 1):
                    print(f"  Pattern #{i}:")
                    print(f"    Confidence: {pattern.confidence:.3f}")
                    print(f"    Combined Score: {pattern.combined_score:.3f}")
                    print(f"    Entry Price: ${pattern.entry_price:.2f}")
                    print(f"    Target Price: ${pattern.target_price:.2f}")
                    print(f"    Stop Loss: ${pattern.stop_loss:.2f}")
                    print(f"    Risk/Reward: {pattern.risk_reward_ratio:.2f}")
                    print(f"    Volume Confirmed: {'✅' if pattern.volume_confirmation else '❌'}")
                    print(f"    Direction: {'🟢 Bullish' if pattern.is_bullish else '🔴 Bearish'}")
                    print(f"    Status: {pattern.status.title()}")
                    
                    # Calculate additional metrics using pattern scorer
                    market_data = pattern_engine.data_manager.preprocess_for_analysis(
                        pattern_engine.data_manager.fetch_stock_data(symbol, "1y"), symbol
                    )
                    
                    if market_data:
                        # Calculate alternative targets
                        fib_target = pattern_scorer.calculate_target_price(pattern, market_data, "fibonacci")
                        sr_target = pattern_scorer.calculate_target_price(pattern, market_data, "support_resistance")
                        
                        # Calculate alternative stops
                        atr_stop = pattern_scorer.calculate_stop_loss(pattern, market_data, "atr")
                        
                        # Pattern quality score
                        quality_score = pattern_scorer.score_pattern_quality(pattern, market_data)
                        
                        print(f"    Quality Score: {quality_score:.3f}")
                        print(f"    Alt Targets: Fib=${fib_target:.2f}, S/R=${sr_target:.2f}")
                        print(f"    Alt Stop (ATR): ${atr_stop:.2f}")
                        
                        # Position sizing example
                        position_info = pattern_scorer.optimize_position_size(
                            pattern, account_balance=100000, max_risk_percent=0.02
                        )
                        print(f"    Position Size: {position_info['shares']} shares (${position_info['position_value']:.0f})")
                        print(f"    Risk Amount: ${position_info['risk_amount']:.0f} ({position_info['risk_percent']:.1f}%)")
        else:
            print(f"❌ No patterns detected in {symbol} with current settings")
    
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
    
    # Show engine statistics
    stats = pattern_engine.get_engine_stats()
    print(f"\n📈 Engine Statistics:")
    print(f"  Total Scans: {stats.get('total_scans', 0)}")
    print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
    print(f"  Patterns Detected: {stats.get('patterns_detected', 0)}")
    print(f"  Patterns Validated: {stats.get('patterns_validated', 0)}")
    if stats.get('avg_detection_time'):
        print(f"  Avg Detection Time: {stats['avg_detection_time']:.2f}s")


def demo_batch_detection():
    """Demonstrate batch pattern detection for multiple stocks."""
    print("\n" + "="*60)
    print("DEMO: Batch Pattern Detection")
    print("="*60)
    
    # Create configuration
    config = PatternConfig(
        min_confidence=0.65,
        min_combined_score=0.7,
        enable_parallel_processing=True,
        max_concurrent_requests=5
    )
    
    # Initialize engine
    pattern_engine = PatternEngine(config)
    
    # Test symbols (major tech stocks)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    print(f"Analyzing {len(symbols)} symbols: {', '.join(symbols)}")
    
    try:
        # Run batch detection
        start_time = datetime.now()
        results = pattern_engine.detect_patterns_batch(
            symbols=symbols,
            period="6mo",
            pattern_types=["Head and Shoulders", "Double Bottom", "Cup and Handle"]
        )
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n⏱️  Batch processing completed in {processing_time:.2f} seconds")
        print(f"📊 Results Summary:")
        
        if results:
            # Generate summary
            summary = pattern_engine.get_detection_summary(results)
            
            print(f"  Symbols with Patterns: {len(results)}/{len(symbols)}")
            print(f"  Total Patterns Found: {summary['total_patterns']}")
            print(f"  Avg Confidence: {summary['confidence_stats']['mean']:.3f}")
            print(f"  Avg Risk/Reward: {summary['risk_reward_stats']['mean']:.2f}")
            print(f"  Volume Confirmation Rate: {summary['volume_confirmation_rate']:.1%}")
            print(f"  Bullish Patterns: {summary['bullish_patterns']}")
            print(f"  Bearish Patterns: {summary['bearish_patterns']}")
            
            print(f"\n📋 Pattern Distribution:")
            for pattern_type, count in summary['patterns_by_type'].items():
                print(f"  {pattern_type}: {count}")
            
            # Show top patterns
            ranked_patterns = pattern_engine.rank_patterns(results, "combined_score")
            
            print(f"\n🏆 Top 5 Patterns by Combined Score:")
            for i, pattern in enumerate(ranked_patterns[:5], 1):
                direction = "🟢 Bullish" if pattern.is_bullish else "🔴 Bearish"
                print(f"  {i}. {pattern.symbol} - {pattern.type} ({direction})")
                print(f"     Score: {pattern.combined_score:.3f}, R/R: {pattern.risk_reward_ratio:.2f}")
        else:
            print("  No patterns detected with current settings")
    
    except Exception as e:
        print(f"❌ Error in batch detection: {e}")


def demo_pattern_filtering():
    """Demonstrate pattern filtering capabilities."""
    print("\n" + "="*60)
    print("DEMO: Pattern Filtering")
    print("="*60)
    
    # Create configuration
    config = PatternConfig(min_confidence=0.5)  # Lower threshold for demo
    pattern_engine = PatternEngine(config)
    
    # Get some patterns first
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = pattern_engine.detect_patterns_batch(symbols, "1y")
    
    if not results:
        print("No patterns found for filtering demo")
        return
    
    print(f"Original results: {pattern_engine.get_detection_summary(results)['total_patterns']} patterns")
    
    # Apply various filters
    filters = [
        {"min_confidence": 0.7, "name": "High Confidence (≥0.7)"},
        {"min_risk_reward": 2.0, "name": "High Risk/Reward (≥2.0)"},
        {"require_volume_confirmation": True, "name": "Volume Confirmed Only"},
        {"pattern_direction": "bullish", "name": "Bullish Patterns Only"},
        {"status": "confirmed", "name": "Confirmed Patterns Only"}
    ]
    
    for filter_config in filters:
        filter_name = filter_config.pop("name")
        filtered_results = pattern_engine.filter_patterns(results, filter_config)
        filtered_count = sum(len(patterns) for patterns in filtered_results.values())
        
        print(f"  {filter_name}: {filtered_count} patterns")
        
        # Add the name back for next iteration
        filter_config["name"] = filter_name


def demo_performance_optimization():
    """Demonstrate Apple Silicon performance optimizations."""
    print("\n" + "="*60)
    print("DEMO: Performance Optimization")
    print("="*60)
    
    # Test different configurations
    configs = [
        {"name": "Sequential Processing", "parallel": False, "workers": 1},
        {"name": "Parallel Processing", "parallel": True, "workers": 4},
        {"name": "High Concurrency", "parallel": True, "workers": 8}
    ]
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
    
    for config_info in configs:
        print(f"\n🔧 Testing {config_info['name']}...")
        
        config = PatternConfig(
            enable_parallel_processing=config_info["parallel"],
            max_concurrent_requests=config_info["workers"]
        )
        
        pattern_engine = PatternEngine(config)
        
        start_time = datetime.now()
        results = pattern_engine.detect_patterns_batch(symbols, "6mo")
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        patterns_found = sum(len(patterns) for symbol_results in results.values() 
                           for patterns in symbol_results.values())
        
        print(f"  Time: {processing_time:.2f}s")
        print(f"  Patterns: {patterns_found}")
        print(f"  Speed: {len(symbols)/processing_time:.1f} symbols/sec")
        
        # Show cache statistics
        cache_stats = pattern_engine.get_engine_stats().get('cache_stats', {})
        if cache_stats.get('size', 0) > 0:
            print(f"  Cache: {cache_stats['size']} items")


def main():
    """Run all demos."""
    print("🚀 Advanced Pattern Scanner - Main Engine Demo")
    print("This demo showcases the complete pattern detection system")
    
    try:
        # Run individual demos
        demo_single_stock_detection()
        demo_batch_detection()
        demo_pattern_filtering()
        demo_performance_optimization()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        
        print("\n🎯 Next Steps:")
        print("1. Run the Streamlit UI: streamlit run ui/streamlit_app.py")
        print("2. Try batch scanning: python batch_scanner.py --symbols 'AAPL,MSFT,GOOGL'")
        print("3. Customize configuration in PatternConfig for your needs")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)


if __name__ == "__main__":
    main()