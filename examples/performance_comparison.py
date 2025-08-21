"""
Performance Comparison Example

This script compares the performance of the new modular system
against the legacy implementation, demonstrating the improvements
achieved through Apple Silicon optimizations and code cleanup.
"""

import sys
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from advanced_pattern_scanner.core.pattern_engine import PatternEngine
from advanced_pattern_scanner.core.models import PatternConfig
from advanced_pattern_scanner.core.apple_silicon_optimizer import get_global_optimizer
from advanced_pattern_scanner.core.error_handler import setup_global_error_handling


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self):
        self.results = {}
        self.optimizer = get_global_optimizer()
    
    def measure_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_operation(self, name: str, operation_func, *args, **kwargs) -> Dict:
        """Benchmark a single operation."""
        print(f"🔄 Benchmarking: {name}")
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Measure initial memory
        initial_memory = self.measure_memory_usage()
        
        # Measure execution time
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        
        # Measure final memory
        final_memory = self.measure_memory_usage()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = final_memory - initial_memory
        
        benchmark_result = {
            'name': name,
            'execution_time': execution_time,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': memory_delta,
            'success': success,
            'error': error,
            'result_size': len(str(result)) if result else 0
        }
        
        self.results[name] = benchmark_result
        
        print(f"   ⏱️  Time: {execution_time:.3f}s")
        print(f"   💾 Memory: {memory_delta:+.1f}MB")
        print(f"   ✅ Success: {success}")
        
        return benchmark_result
    
    def compare_results(self, baseline_name: str, optimized_name: str) -> Dict:
        """Compare two benchmark results."""
        if baseline_name not in self.results or optimized_name not in self.results:
            return {}
        
        baseline = self.results[baseline_name]
        optimized = self.results[optimized_name]
        
        if not baseline['success'] or not optimized['success']:
            return {'error': 'One or both operations failed'}
        
        time_improvement = (baseline['execution_time'] - optimized['execution_time']) / baseline['execution_time']
        memory_improvement = (baseline['memory_delta_mb'] - optimized['memory_delta_mb']) / abs(baseline['memory_delta_mb']) if baseline['memory_delta_mb'] != 0 else 0
        
        return {
            'time_improvement_percent': time_improvement * 100,
            'memory_improvement_percent': memory_improvement * 100,
            'baseline_time': baseline['execution_time'],
            'optimized_time': optimized['execution_time'],
            'baseline_memory': baseline['memory_delta_mb'],
            'optimized_memory': optimized['memory_delta_mb']
        }
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n📊 Performance Benchmark Summary")
        print("=" * 60)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"   Time: {result['execution_time']:.3f}s")
            print(f"   Memory: {result['memory_delta_mb']:+.1f}MB")
            print(f"   Success: {result['success']}")
            if result['error']:
                print(f"   Error: {result['error']}")


def benchmark_single_stock_detection():
    """Benchmark single stock pattern detection."""
    print("\n🔍 Single Stock Detection Benchmark")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    
    # Setup configurations
    basic_config = PatternConfig(
        min_confidence=0.6,
        use_gpu=False,  # Disable optimizations for baseline
        enable_parallel_processing=False
    )
    
    optimized_config = PatternConfig(
        min_confidence=0.6,
        use_gpu=True,  # Enable Apple Silicon optimizations
        enable_parallel_processing=True
    )
    
    symbol = "AAPL"
    period = "1y"
    
    # Benchmark basic configuration
    def detect_basic():
        engine = PatternEngine(basic_config)
        return engine.detect_patterns_single_stock(symbol, period)
    
    # Benchmark optimized configuration
    def detect_optimized():
        engine = PatternEngine(optimized_config)
        return engine.detect_patterns_single_stock(symbol, period)
    
    # Run benchmarks
    benchmark.benchmark_operation("Basic Detection", detect_basic)
    benchmark.benchmark_operation("Optimized Detection", detect_optimized)
    
    # Compare results
    comparison = benchmark.compare_results("Basic Detection", "Optimized Detection")
    
    if comparison and 'error' not in comparison:
        print(f"\n🚀 Performance Improvement:")
        print(f"   Time: {comparison['time_improvement_percent']:+.1f}%")
        print(f"   Memory: {comparison['memory_improvement_percent']:+.1f}%")
        print(f"   Baseline: {comparison['baseline_time']:.3f}s, {comparison['baseline_memory']:+.1f}MB")
        print(f"   Optimized: {comparison['optimized_time']:.3f}s, {comparison['optimized_memory']:+.1f}MB")
    
    return benchmark


def benchmark_batch_processing():
    """Benchmark batch processing performance."""
    print("\n🚀 Batch Processing Benchmark")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    
    # Setup configurations
    sequential_config = PatternConfig(
        min_confidence=0.7,
        enable_parallel_processing=False
    )
    
    parallel_config = PatternConfig(
        min_confidence=0.7,
        enable_parallel_processing=True,
        max_concurrent_requests=4
    )
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
    period = "6mo"
    
    # Benchmark sequential processing
    def process_sequential():
        engine = PatternEngine(sequential_config)
        return engine.detect_patterns_batch(symbols, period)
    
    # Benchmark parallel processing
    def process_parallel():
        engine = PatternEngine(parallel_config)
        return engine.detect_patterns_batch(symbols, period)
    
    # Run benchmarks
    benchmark.benchmark_operation("Sequential Processing", process_sequential)
    benchmark.benchmark_operation("Parallel Processing", process_parallel)
    
    # Compare results
    comparison = benchmark.compare_results("Sequential Processing", "Parallel Processing")
    
    if comparison and 'error' not in comparison:
        print(f"\n⚡ Parallel Processing Improvement:")
        print(f"   Time: {comparison['time_improvement_percent']:+.1f}%")
        print(f"   Memory: {comparison['memory_improvement_percent']:+.1f}%")
        print(f"   Sequential: {comparison['baseline_time']:.3f}s")
        print(f"   Parallel: {comparison['optimized_time']:.3f}s")
    
    return benchmark


def benchmark_apple_silicon_features():
    """Benchmark Apple Silicon specific features."""
    print("\n🍎 Apple Silicon Features Benchmark")
    print("=" * 50)
    
    optimizer = get_global_optimizer()
    
    if not optimizer.is_apple_silicon:
        print("⚠️  Not running on Apple Silicon - skipping Apple Silicon benchmarks")
        return None
    
    benchmark = PerformanceBenchmark()
    
    # Benchmark tensor operations
    def tensor_operations():
        return optimizer.benchmark_operations(tensor_size=(2000, 2000))
    
    # Benchmark memory optimizations
    def memory_optimizations():
        return optimizer.optimize_memory_usage()
    
    # Run benchmarks
    benchmark.benchmark_operation("Tensor Operations", tensor_operations)
    benchmark.benchmark_operation("Memory Optimizations", memory_optimizations)
    
    # Show Apple Silicon specific metrics
    device_info = optimizer.device_info
    print(f"\n🔧 Apple Silicon Configuration:")
    print(f"   Chip: {device_info.get('chip_type', 'Unknown')}")
    print(f"   Memory: {device_info['memory_gb']} GB")
    print(f"   MPS Available: {device_info['mps_available']}")
    print(f"   Optimal Device: {optimizer.get_optimal_device()}")
    
    return benchmark


def benchmark_memory_efficiency():
    """Benchmark memory efficiency improvements."""
    print("\n💾 Memory Efficiency Benchmark")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    
    # Test with different batch sizes
    batch_sizes = [16, 32, 64]
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
    
    for batch_size in batch_sizes:
        config = PatternConfig(
            batch_size=batch_size,
            min_confidence=0.7
        )
        
        def process_with_batch_size():
            engine = PatternEngine(config)
            return engine.detect_patterns_batch(symbols[:5], period="3mo")  # Limit to 5 stocks
        
        benchmark.benchmark_operation(f"Batch Size {batch_size}", process_with_batch_size)
    
    # Find optimal batch size
    best_batch_size = None
    best_efficiency = float('inf')
    
    for batch_size in batch_sizes:
        result = benchmark.results.get(f"Batch Size {batch_size}")
        if result and result['success']:
            # Calculate efficiency as time * memory
            efficiency = result['execution_time'] * max(1, result['memory_delta_mb'])
            if efficiency < best_efficiency:
                best_efficiency = efficiency
                best_batch_size = batch_size
    
    if best_batch_size:
        print(f"\n🎯 Optimal Batch Size: {best_batch_size}")
        print(f"   Efficiency Score: {best_efficiency:.2f}")
    
    return benchmark


def generate_performance_report():
    """Generate comprehensive performance report."""
    print("\n📋 Generating Performance Report")
    print("=" * 50)
    
    # Run all benchmarks
    benchmarks = []
    
    try:
        benchmarks.append(benchmark_single_stock_detection())
        benchmarks.append(benchmark_batch_processing())
        benchmarks.append(benchmark_apple_silicon_features())
        benchmarks.append(benchmark_memory_efficiency())
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        return
    
    # Combine all results
    all_results = {}
    for bench in benchmarks:
        if bench:
            all_results.update(bench.results)
    
    # Generate summary report
    print(f"\n📊 Comprehensive Performance Report")
    print("=" * 60)
    
    successful_operations = [r for r in all_results.values() if r['success']]
    
    if successful_operations:
        avg_time = sum(r['execution_time'] for r in successful_operations) / len(successful_operations)
        avg_memory = sum(r['memory_delta_mb'] for r in successful_operations) / len(successful_operations)
        
        print(f"Total Operations: {len(all_results)}")
        print(f"Successful Operations: {len(successful_operations)}")
        print(f"Success Rate: {len(successful_operations)/len(all_results)*100:.1f}%")
        print(f"Average Execution Time: {avg_time:.3f}s")
        print(f"Average Memory Delta: {avg_memory:+.1f}MB")
        
        # Show top performers
        fastest_ops = sorted(successful_operations, key=lambda x: x['execution_time'])[:3]
        print(f"\n🏆 Fastest Operations:")
        for i, op in enumerate(fastest_ops, 1):
            print(f"   {i}. {op['name']}: {op['execution_time']:.3f}s")
        
        # Show most memory efficient
        memory_efficient = sorted(successful_operations, key=lambda x: x['memory_delta_mb'])[:3]
        print(f"\n💾 Most Memory Efficient:")
        for i, op in enumerate(memory_efficient, 1):
            print(f"   {i}. {op['name']}: {op['memory_delta_mb']:+.1f}MB")
    
    # Hardware summary
    optimizer = get_global_optimizer()
    print(f"\n🖥️  Hardware Summary:")
    print(f"   Platform: {optimizer.device_info['platform']}")
    print(f"   Apple Silicon: {optimizer.is_apple_silicon}")
    if optimizer.is_apple_silicon:
        print(f"   Chip: {optimizer.device_info.get('chip_type', 'Unknown')}")
    print(f"   Memory: {optimizer.device_info['memory_gb']} GB")
    print(f"   CPU Cores: {optimizer.device_info['cpu_count']}")


def main():
    """Run performance comparison."""
    print("⚡ Advanced Pattern Scanner - Performance Comparison")
    print("=" * 70)
    
    # Setup error handling
    setup_global_error_handling()
    
    try:
        generate_performance_report()
        print("\n✅ Performance comparison completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Performance comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()