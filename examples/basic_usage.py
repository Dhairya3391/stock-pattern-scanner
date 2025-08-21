"""
Basic Usage Examples for Advanced Pattern Scanner

This script demonstrates the basic functionality of the new modular
pattern detection system with Apple Silicon optimizations.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import the scanner
sys.path.append(str(Path(__file__).parent.parent))

from advanced_pattern_scanner.core.pattern_engine import PatternEngine
from advanced_pattern_scanner.core.models import PatternConfig
from advanced_pattern_scanner.core.apple_silicon_optimizer import get_global_optimizer
from advanced_pattern_scanner.core.error_handler import setup_global_error_handling


def basic_pattern_detection():
    """Demonstrate basic pattern detection for a single stock."""
    print("🔍 Basic Pattern Detection Example")
    print("=" * 50)
    
    # Setup error handling
    setup_global_error_handling()
    
    # Initialize configuration
    config = PatternConfig(
        min_confidence=0.6,
        volume_confirmation_required=True,
        use_gpu=True  # Enable Apple Silicon optimizations
    )
    
    # Create pattern engine
    engine = PatternEngine(config)
    
    # Detect patterns for Apple stock
    symbol = "AAPL"
    print(f"Analyzing {symbol}...")
    
    try:
        patterns = engine.detect_patterns_single_stock(symbol, period="1y")
        
        if patterns:
            print(f"\n✅ Found patterns for {symbol}:")
            
            for pattern_type, pattern_list in patterns.items():
                print(f"\n📊 {pattern_type}: {len(pattern_list)} patterns")
                
                for i, pattern in enumerate(pattern_list[:3], 1):  # Show top 3
                    print(f"  {i}. Confidence: {pattern.confidence:.3f}")
                    print(f"     Target Price: ${pattern.target_price:.2f}")
                    print(f"     Risk/Reward: {pattern.risk_reward_ratio:.2f}")
                    print(f"     Status: {pattern.status}")
                    print(f"     Volume Confirmed: {pattern.volume_confirmation}")
        else:
            print(f"❌ No patterns found for {symbol}")
            
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
    
    # Show engine statistics
    stats = engine.get_engine_stats()
    print(f"\n📈 Engine Statistics:")
    print(f"   Total scans: {stats['total_scans']}")
    print(f"   Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"   Patterns detected: {stats['patterns_detected']}")
    print(f"   Patterns validated: {stats['patterns_validated']}")


def batch_processing_example():
    """Demonstrate batch processing of multiple stocks."""
    print("\n🚀 Batch Processing Example")
    print("=" * 50)
    
    # Initialize configuration for batch processing
    config = PatternConfig(
        min_confidence=0.7,
        enable_parallel_processing=True,
        max_concurrent_requests=4
    )
    
    engine = PatternEngine(config)
    
    # Define watchlist
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    print(f"Analyzing {len(symbols)} stocks: {', '.join(symbols)}")
    
    try:
        # Batch process all symbols
        batch_results = engine.detect_patterns_batch(symbols, period="6mo")
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Stocks with patterns: {len(batch_results)}")
        
        # Rank all patterns across stocks
        all_patterns = []
        for symbol, patterns in batch_results.items():
            for pattern_list in patterns.values():
                all_patterns.extend(pattern_list)
        
        # Sort by combined score
        top_patterns = sorted(all_patterns, key=lambda p: p.combined_score, reverse=True)[:10]
        
        print(f"\n🏆 Top 10 Patterns Across All Stocks:")
        for i, pattern in enumerate(top_patterns, 1):
            print(f"  {i:2d}. {pattern.symbol} - {pattern.type}")
            print(f"      Score: {pattern.combined_score:.3f} | "
                  f"Confidence: {pattern.confidence:.3f} | "
                  f"R/R: {pattern.risk_reward_ratio:.2f}")
        
        # Generate summary
        summary = engine.get_detection_summary(batch_results)
        print(f"\n📊 Detection Summary:")
        print(f"   Total patterns: {summary['total_patterns']}")
        print(f"   Average confidence: {summary['confidence_stats']['mean']:.3f}")
        print(f"   Bullish patterns: {summary['bullish_patterns']}")
        print(f"   Bearish patterns: {summary['bearish_patterns']}")
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")


def apple_silicon_optimization_demo():
    """Demonstrate Apple Silicon optimization features."""
    print("\n🍎 Apple Silicon Optimization Demo")
    print("=" * 50)
    
    # Get optimizer instance
    optimizer = get_global_optimizer()
    
    print(f"Hardware Information:")
    print(f"   Apple Silicon: {optimizer.is_apple_silicon}")
    print(f"   Platform: {optimizer.device_info['platform']}")
    print(f"   Machine: {optimizer.device_info['machine']}")
    print(f"   CPU Count: {optimizer.device_info['cpu_count']}")
    print(f"   Memory: {optimizer.device_info['memory_gb']} GB")
    print(f"   MPS Available: {optimizer.device_info['mps_available']}")
    
    if optimizer.is_apple_silicon:
        print(f"   Chip Type: {optimizer.device_info['chip_type']}")
        print(f"   Unified Memory: {optimizer.device_info['unified_memory']}")
    
    # Get performance recommendations
    recommendations = optimizer.get_performance_recommendations()
    print(f"\n⚡ Performance Recommendations:")
    print(f"   Optimal Device: {recommendations['device']}")
    print(f"   Recommended Batch Size: {recommendations['optimal_batch_size']}")
    print(f"   Enable AMP: {recommendations['enable_amp']}")
    print(f"   Use Vectorized Ops: {recommendations['use_vectorized_ops']}")
    
    if optimizer.is_apple_silicon:
        threading = recommendations['threading']
        print(f"   Torch Threads: {threading['torch_threads']}")
        print(f"   NumPy Threads: {threading['numpy_threads']}")
    
    # Run benchmark
    print(f"\n🏃 Performance Benchmark:")
    benchmark_results = optimizer.benchmark_operations(tensor_size=(1000, 1000))
    
    if 'error' not in benchmark_results:
        print(f"   Device: {benchmark_results['device']}")
        print(f"   Matrix Multiplication: {benchmark_results['matrix_multiplication_ms']:.2f}ms")
        print(f"   Element-wise Operations: {benchmark_results['elementwise_ops_ms']:.2f}ms")
        
        if 'memory_transfer_ms' in benchmark_results:
            print(f"   Memory Transfer: {benchmark_results['memory_transfer_ms']:.2f}ms")
    else:
        print(f"   Benchmark failed: {benchmark_results['error']}")


def pattern_filtering_example():
    """Demonstrate pattern filtering capabilities."""
    print("\n🔍 Pattern Filtering Example")
    print("=" * 50)
    
    config = PatternConfig(min_confidence=0.5)  # Lower threshold to get more patterns
    engine = PatternEngine(config)
    
    # Get patterns for a stock
    symbol = "TSLA"
    patterns = engine.detect_patterns_single_stock(symbol, period="1y")
    
    if patterns:
        # Count original patterns
        original_count = sum(len(pattern_list) for pattern_list in patterns.values())
        print(f"Original patterns found for {symbol}: {original_count}")
        
        # Apply various filters
        filters = {
            "min_confidence": 0.75,
            "min_risk_reward": 2.0,
            "require_volume_confirmation": True,
            "pattern_direction": "bullish"
        }
        
        filtered_patterns = engine.filter_patterns(patterns, filters)
        filtered_count = sum(len(pattern_list) for pattern_list in filtered_patterns.values())
        
        print(f"\n🎯 After applying filters:")
        print(f"   Minimum confidence: {filters['min_confidence']}")
        print(f"   Minimum risk/reward: {filters['min_risk_reward']}")
        print(f"   Volume confirmation required: {filters['require_volume_confirmation']}")
        print(f"   Direction: {filters['pattern_direction']}")
        print(f"   Filtered patterns: {filtered_count}")
        
        if filtered_patterns:
            print(f"\n✅ High-quality patterns:")
            for pattern_type, pattern_list in filtered_patterns.items():
                for pattern in pattern_list:
                    print(f"   {pattern_type}: Confidence {pattern.confidence:.3f}, "
                          f"R/R {pattern.risk_reward_ratio:.2f}")
    else:
        print(f"No patterns found for {symbol}")


def main():
    """Run all examples."""
    print("🚀 Advanced Pattern Scanner - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        basic_pattern_detection()
        batch_processing_example()
        apple_silicon_optimization_demo()
        pattern_filtering_example()
        
        print("\n✅ All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()