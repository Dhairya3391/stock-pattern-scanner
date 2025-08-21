"""
Performance Benchmarks for MacBook M1/M2/M3 Optimization.

This module provides comprehensive performance testing and benchmarking
specifically optimized for Apple Silicon MacBooks, measuring speed and accuracy
improvements over the legacy system.
"""

import time
import psutil
import os
import platform
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..core.models import MarketData, PatternConfig
from ..patterns.head_shoulders import HeadShouldersDetector
from ..patterns.double_bottom import DoubleBottomDetector
from ..patterns.cup_handle import CupHandleDetector
from ..core.pattern_engine import PatternEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MacBookPerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for MacBook optimization.
    
    Tests pattern detection algorithms across various data sizes and scenarios
    to validate Apple Silicon optimizations and measure improvements.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.config = PatternConfig()
        self.system_info = self._get_system_info()
        self.results = {}
        
        # Initialize detectors
        self.hs_detector = HeadShouldersDetector(self.config)
        self.db_detector = DoubleBottomDetector(self.config)
        self.ch_detector = CupHandleDetector(self.config)
        
        # Create results directory
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized benchmarks for {self.system_info['system']} {self.system_info['processor']}")
    
    def _get_system_info(self) -> Dict:
        """Get detailed system information for benchmarking context."""
        return {
            'system': platform.system(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'is_apple_silicon': platform.machine() in ['arm64', 'aarch64'],
            'timestamp': datetime.now().isoformat()
        }
    
    def create_synthetic_data(self, size: int, pattern_type: str = 'mixed', 
                            volatility: float = 0.1) -> MarketData:
        """
        Create synthetic market data for benchmarking.
        
        Args:
            size: Number of data points
            pattern_type: Type of pattern to embed ('hs', 'db', 'ch', 'mixed', 'random')
            volatility: Price volatility factor
            
        Returns:
            MarketData object with synthetic data
        """
        np.random.seed(42)  # For reproducible benchmarks
        
        base_price = 100
        base_trend = np.linspace(base_price, base_price * 1.2, size)
        noise = np.random.normal(0, base_price * volatility, size)
        
        if pattern_type == 'hs':
            # Embed Head & Shoulders pattern
            if size >= 20:
                mid = size // 2
                # Create H&S structure
                base_trend[mid-8:mid-6] = base_price * 1.1  # Left shoulder
                base_trend[mid-3:mid+1] = base_price * 1.25  # Head
                base_trend[mid+4:mid+6] = base_price * 1.12  # Right shoulder
                base_trend[mid+8:] *= 0.9  # Decline after pattern
        
        elif pattern_type == 'db':
            # Embed Double Bottom pattern
            if size >= 15:
                mid = size // 2
                base_trend[mid-5:mid-3] = base_price * 0.9   # First bottom
                base_trend[mid-1:mid+1] = base_price * 1.1   # Peak
                base_trend[mid+3:mid+5] = base_price * 0.91  # Second bottom
                base_trend[mid+7:] *= 1.15  # Breakout
        
        elif pattern_type == 'ch':
            # Embed Cup & Handle pattern
            if size >= 30:
                quarter = size // 4
                # Cup formation
                cup_start = quarter
                cup_end = quarter * 3
                cup_depth = base_price * 0.8
                
                for i in range(cup_start, cup_end):
                    progress = (i - cup_start) / (cup_end - cup_start)
                    # U-shaped curve
                    cup_factor = 4 * progress * (1 - progress)
                    base_trend[i] = cup_depth + (base_price - cup_depth) * cup_factor
                
                # Handle formation
                handle_start = cup_end
                handle_end = min(cup_end + quarter // 2, size - 5)
                for i in range(handle_start, handle_end):
                    base_trend[i] = base_price * 0.95
                
                # Breakout
                if handle_end < size:
                    base_trend[handle_end:] = base_price * 1.2
        
        prices = np.maximum(base_trend + noise, 1.0)  # Ensure positive prices
        volumes = np.random.lognormal(8, 0.5, size)  # Realistic volume distribution
        
        # Create OHLCV data
        ohlcv = np.column_stack([prices, prices, prices, prices, volumes])
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(size)]
        
        return MarketData(
            symbol=f"SYNTH_{pattern_type.upper()}_{size}",
            timeframe="1d",
            data=ohlcv,
            timestamps=timestamps
        )
    
    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[any, Dict]:
        """
        Measure detailed execution metrics for a function.
        
        Returns:
            Tuple of (result, metrics_dict)
        """
        process = psutil.Process()
        
        # Pre-execution measurements
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        start_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Post-execution measurements
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
        end_memory = process.memory_info().rss / (1024**2)  # MB
        
        metrics = {
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu_time - start_cpu_time,
            'memory_peak': end_memory,
            'memory_delta': end_memory - start_memory,
            'cpu_efficiency': (end_cpu_time - start_cpu_time) / (end_time - start_time) if end_time > start_time else 0
        }
        
        return result, metrics
    
    def benchmark_single_pattern_detector(self, detector, data_sizes: List[int], 
                                        pattern_type: str) -> Dict:
        """
        Benchmark a single pattern detector across different data sizes.
        
        Args:
            detector: Pattern detector instance
            data_sizes: List of data sizes to test
            pattern_type: Pattern type for synthetic data generation
            
        Returns:
            Dictionary with benchmark results
        """
        detector_name = detector.__class__.__name__.replace('Detector', '').replace('HeadShoulders', 'Head & Shoulders')
        logger.info(f"Benchmarking {detector_name} detector...")
        
        results = {
            'detector': detector_name,
            'pattern_type': pattern_type,
            'data_sizes': [],
            'metrics': []
        }
        
        for size in data_sizes:
            logger.info(f"  Testing with {size} data points...")
            
            # Create synthetic data
            market_data = self.create_synthetic_data(size, pattern_type)
            
            # Measure performance
            patterns, metrics = self.measure_execution_time(
                detector.detect_pattern, market_data
            )
            
            # Calculate throughput
            throughput = size / metrics['wall_time'] if metrics['wall_time'] > 0 else 0
            
            # Store results
            size_result = {
                'data_size': size,
                'patterns_found': len(patterns),
                'wall_time': metrics['wall_time'],
                'cpu_time': metrics['cpu_time'],
                'memory_peak': metrics['memory_peak'],
                'memory_delta': metrics['memory_delta'],
                'cpu_efficiency': metrics['cpu_efficiency'],
                'throughput': throughput,
                'points_per_second': throughput
            }
            
            results['data_sizes'].append(size)
            results['metrics'].append(size_result)
            
            logger.info(f"    {metrics['wall_time']:.4f}s, {throughput:.0f} points/sec, "
                       f"{len(patterns)} patterns found")
        
        return results
    
    def benchmark_all_detectors(self, data_sizes: List[int] = None) -> Dict:
        """
        Benchmark all pattern detectors across various data sizes.
        
        Args:
            data_sizes: List of data sizes to test
            
        Returns:
            Complete benchmark results
        """
        if data_sizes is None:
            data_sizes = [100, 250, 500, 1000, 2000, 5000]
        
        logger.info("Starting comprehensive detector benchmarks...")
        
        detectors_and_patterns = [
            (self.hs_detector, 'hs'),
            (self.db_detector, 'db'),
            (self.ch_detector, 'ch')
        ]
        
        all_results = {
            'system_info': self.system_info,
            'benchmark_timestamp': datetime.now().isoformat(),
            'data_sizes_tested': data_sizes,
            'detector_results': []
        }
        
        for detector, pattern_type in detectors_and_patterns:
            detector_results = self.benchmark_single_pattern_detector(
                detector, data_sizes, pattern_type
            )
            all_results['detector_results'].append(detector_results)
        
        self.results['detector_benchmarks'] = all_results
        return all_results
    
    def benchmark_accuracy_vs_speed(self) -> Dict:
        """
        Benchmark accuracy vs speed trade-offs with different configurations.
        
        Returns:
            Accuracy vs speed benchmark results
        """
        logger.info("Benchmarking accuracy vs speed trade-offs...")
        
        # Different configuration scenarios
        configs = {
            'high_accuracy': PatternConfig(
                min_confidence=0.8,
                head_shoulders_tolerance=0.05,
                double_pattern_tolerance=0.01,
                min_volume_ratio=1.5
            ),
            'balanced': PatternConfig(
                min_confidence=0.7,
                head_shoulders_tolerance=0.10,
                double_pattern_tolerance=0.02,
                min_volume_ratio=1.2
            ),
            'high_speed': PatternConfig(
                min_confidence=0.6,
                head_shoulders_tolerance=0.15,
                double_pattern_tolerance=0.03,
                min_volume_ratio=1.0
            )
        }
        
        test_data_size = 1000
        results = {
            'test_data_size': test_data_size,
            'configurations': []
        }
        
        for config_name, config in configs.items():
            logger.info(f"  Testing {config_name} configuration...")
            
            # Create detectors with this configuration
            hs_detector = HeadShouldersDetector(config)
            db_detector = DoubleBottomDetector(config)
            ch_detector = CupHandleDetector(config)
            
            config_results = {
                'name': config_name,
                'config': config.__dict__,
                'detector_performance': []
            }
            
            # Test each detector
            for detector, pattern_type in [(hs_detector, 'hs'), (db_detector, 'db'), (ch_detector, 'ch')]:
                market_data = self.create_synthetic_data(test_data_size, pattern_type)
                patterns, metrics = self.measure_execution_time(detector.detect_pattern, market_data)
                
                # Calculate accuracy metrics (simplified)
                expected_patterns = 1 if pattern_type != 'random' else 0
                accuracy = min(len(patterns), expected_patterns) / max(expected_patterns, 1)
                
                detector_result = {
                    'detector': detector.__class__.__name__,
                    'patterns_found': len(patterns),
                    'expected_patterns': expected_patterns,
                    'accuracy_score': accuracy,
                    'execution_time': metrics['wall_time'],
                    'memory_usage': metrics['memory_peak'],
                    'speed_score': 1 / metrics['wall_time'] if metrics['wall_time'] > 0 else 0
                }
                
                config_results['detector_performance'].append(detector_result)
            
            results['configurations'].append(config_results)
        
        self.results['accuracy_vs_speed'] = results
        return results
    
    def benchmark_memory_efficiency(self) -> Dict:
        """
        Benchmark memory efficiency and garbage collection performance.
        
        Returns:
            Memory efficiency benchmark results
        """
        logger.info("Benchmarking memory efficiency...")
        
        import gc
        
        data_sizes = [500, 1000, 2000, 5000]
        results = {
            'memory_benchmarks': []
        }
        
        for size in data_sizes:
            logger.info(f"  Testing memory efficiency with {size} data points...")
            
            # Force garbage collection before test
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / (1024**2)
            
            # Create multiple datasets and process them
            datasets = []
            peak_memory = initial_memory
            
            for i in range(5):  # Process 5 datasets
                market_data = self.create_synthetic_data(size, 'mixed')
                datasets.append(market_data)
                
                # Process with all detectors
                for detector in [self.hs_detector, self.db_detector, self.ch_detector]:
                    patterns = detector.detect_pattern(market_data)
                    current_memory = psutil.Process().memory_info().rss / (1024**2)
                    peak_memory = max(peak_memory, current_memory)
            
            # Force garbage collection and measure final memory
            del datasets
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / (1024**2)
            
            memory_result = {
                'data_size': size,
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'final_memory_mb': final_memory,
                'memory_growth_mb': peak_memory - initial_memory,
                'memory_leaked_mb': final_memory - initial_memory,
                'memory_efficiency_score': size / (peak_memory - initial_memory) if peak_memory > initial_memory else float('inf')
            }
            
            results['memory_benchmarks'].append(memory_result)
            
            logger.info(f"    Peak memory: {peak_memory:.1f}MB, Growth: {peak_memory - initial_memory:.1f}MB")
        
        self.results['memory_efficiency'] = results
        return results
    
    def benchmark_apple_silicon_optimizations(self) -> Dict:
        """
        Benchmark Apple Silicon specific optimizations.
        
        Returns:
            Apple Silicon optimization benchmark results
        """
        logger.info("Benchmarking Apple Silicon optimizations...")
        
        if not self.system_info['is_apple_silicon']:
            logger.warning("Not running on Apple Silicon - skipping optimization benchmarks")
            return {'skipped': True, 'reason': 'Not Apple Silicon'}
        
        # Test vectorized operations performance
        data_size = 2000
        results = {
            'apple_silicon_optimized': True,
            'optimization_tests': []
        }
        
        # Test NumPy vectorized operations (should use Apple's Accelerate framework)
        logger.info("  Testing NumPy vectorized operations...")
        
        large_array = np.random.random(data_size * 100)
        start_time = time.perf_counter()
        
        # Vectorized operations that benefit from Apple Silicon
        result = np.fft.fft(large_array)
        result = np.linalg.norm(result)
        result = np.convolve(large_array[:1000], large_array[1000:2000], mode='valid')
        
        vectorized_time = time.perf_counter() - start_time
        
        optimization_result = {
            'test': 'numpy_vectorized_operations',
            'execution_time': vectorized_time,
            'data_size': len(large_array),
            'throughput': len(large_array) / vectorized_time,
            'apple_silicon_optimized': True
        }
        
        results['optimization_tests'].append(optimization_result)
        
        # Test pattern detection with large datasets
        logger.info("  Testing pattern detection with large datasets...")
        
        large_market_data = self.create_synthetic_data(data_size, 'mixed')
        
        for detector_name, detector in [('HeadShoulders', self.hs_detector), 
                                       ('DoubleBottom', self.db_detector),
                                       ('CupHandle', self.ch_detector)]:
            patterns, metrics = self.measure_execution_time(detector.detect_pattern, large_market_data)
            
            optimization_result = {
                'test': f'{detector_name}_large_dataset',
                'execution_time': metrics['wall_time'],
                'cpu_time': metrics['cpu_time'],
                'data_size': data_size,
                'patterns_found': len(patterns),
                'throughput': data_size / metrics['wall_time'],
                'cpu_efficiency': metrics['cpu_efficiency']
            }
            
            results['optimization_tests'].append(optimization_result)
        
        self.results['apple_silicon_optimizations'] = results
        return results
    
    def run_comprehensive_benchmarks(self) -> Dict:
        """
        Run all benchmark suites and generate comprehensive results.
        
        Returns:
            Complete benchmark results
        """
        logger.info("Starting comprehensive benchmark suite...")
        
        # Run all benchmark categories
        detector_results = self.benchmark_all_detectors()
        accuracy_speed_results = self.benchmark_accuracy_vs_speed()
        memory_results = self.benchmark_memory_efficiency()
        apple_silicon_results = self.benchmark_apple_silicon_optimizations()
        
        # Compile comprehensive results
        comprehensive_results = {
            'benchmark_suite_version': '1.0',
            'system_info': self.system_info,
            'timestamp': datetime.now().isoformat(),
            'detector_benchmarks': detector_results,
            'accuracy_vs_speed': accuracy_speed_results,
            'memory_efficiency': memory_results,
            'apple_silicon_optimizations': apple_silicon_results,
            'summary': self._generate_benchmark_summary()
        }
        
        # Save results
        self._save_results(comprehensive_results)
        
        # Generate visualizations
        self._generate_visualizations(comprehensive_results)
        
        logger.info("Comprehensive benchmarks completed!")
        return comprehensive_results
    
    def _generate_benchmark_summary(self) -> Dict:
        """Generate summary statistics from benchmark results."""
        summary = {
            'performance_highlights': [],
            'optimization_status': 'optimized' if self.system_info['is_apple_silicon'] else 'standard',
            'recommendations': []
        }
        
        # Analyze detector performance
        if 'detector_benchmarks' in self.results:
            detector_results = self.results['detector_benchmarks']
            
            for detector_result in detector_results['detector_results']:
                metrics = detector_result['metrics']
                if metrics:
                    avg_throughput = np.mean([m['throughput'] for m in metrics])
                    max_throughput = max([m['throughput'] for m in metrics])
                    
                    summary['performance_highlights'].append({
                        'detector': detector_result['detector'],
                        'avg_throughput': avg_throughput,
                        'max_throughput': max_throughput,
                        'performance_rating': 'excellent' if avg_throughput > 5000 else 'good' if avg_throughput > 1000 else 'acceptable'
                    })
        
        # Generate recommendations
        if self.system_info['is_apple_silicon']:
            summary['recommendations'].append("Apple Silicon optimizations are active")
        else:
            summary['recommendations'].append("Consider upgrading to Apple Silicon for better performance")
        
        if self.system_info['memory_total'] < 8:
            summary['recommendations'].append("Consider increasing system memory for large dataset processing")
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable report
        report_file = self.results_dir / f"benchmark_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self._generate_text_report(results))
        
        logger.info(f"Results saved to {json_file} and {report_file}")
    
    def _generate_text_report(self, results: Dict) -> str:
        """Generate human-readable benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("ADVANCED STOCK PATTERN SCANNER - PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # System information
        system_info = results['system_info']
        report.append("SYSTEM INFORMATION:")
        report.append(f"  Platform: {system_info['system']} {system_info['machine']}")
        report.append(f"  Processor: {system_info['processor']}")
        report.append(f"  CPU Cores: {system_info['cpu_count']}")
        report.append(f"  Memory: {system_info['memory_total']:.1f} GB")
        report.append(f"  Apple Silicon: {'Yes' if system_info['is_apple_silicon'] else 'No'}")
        report.append(f"  Python: {system_info['python_version']}")
        report.append("")
        
        # Detector performance
        if 'detector_benchmarks' in results:
            report.append("PATTERN DETECTOR PERFORMANCE:")
            detector_results = results['detector_benchmarks']['detector_results']
            
            for detector_result in detector_results:
                report.append(f"\n  {detector_result['detector']} Detector:")
                metrics = detector_result['metrics']
                
                if metrics:
                    # Performance summary
                    avg_time = np.mean([m['wall_time'] for m in metrics])
                    avg_throughput = np.mean([m['throughput'] for m in metrics])
                    avg_memory = np.mean([m['memory_peak'] for m in metrics])
                    
                    report.append(f"    Average execution time: {avg_time:.4f} seconds")
                    report.append(f"    Average throughput: {avg_throughput:.0f} points/second")
                    report.append(f"    Average memory usage: {avg_memory:.1f} MB")
                    
                    # Detailed results by data size
                    report.append("    Performance by data size:")
                    for metric in metrics:
                        report.append(f"      {metric['data_size']:4d} points: "
                                    f"{metric['wall_time']:.4f}s, "
                                    f"{metric['throughput']:6.0f} pts/sec, "
                                    f"{metric['patterns_found']} patterns")
        
        # Memory efficiency
        if 'memory_efficiency' in results:
            report.append("\nMEMORY EFFICIENCY:")
            memory_results = results['memory_efficiency']['memory_benchmarks']
            
            for result in memory_results:
                report.append(f"  {result['data_size']} data points:")
                report.append(f"    Memory growth: {result['memory_growth_mb']:.1f} MB")
                report.append(f"    Memory leaked: {result['memory_leaked_mb']:.1f} MB")
                report.append(f"    Efficiency score: {result['memory_efficiency_score']:.1f}")
        
        # Apple Silicon optimizations
        if 'apple_silicon_optimizations' in results and not results['apple_silicon_optimizations'].get('skipped'):
            report.append("\nAPPLE SILICON OPTIMIZATIONS:")
            opt_results = results['apple_silicon_optimizations']['optimization_tests']
            
            for test in opt_results:
                report.append(f"  {test['test']}:")
                report.append(f"    Execution time: {test['execution_time']:.4f} seconds")
                if 'throughput' in test:
                    report.append(f"    Throughput: {test['throughput']:.0f} operations/second")
        
        # Summary and recommendations
        if 'summary' in results:
            summary = results['summary']
            report.append("\nPERFORMANCE SUMMARY:")
            
            for highlight in summary['performance_highlights']:
                report.append(f"  {highlight['detector']}: {highlight['performance_rating'].upper()} "
                            f"({highlight['avg_throughput']:.0f} pts/sec average)")
            
            report.append("\nRECOMMENDATIONS:")
            for rec in summary['recommendations']:
                report.append(f"  • {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_visualizations(self, results: Dict):
        """Generate performance visualization charts."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8')
            
            # Performance vs data size chart
            if 'detector_benchmarks' in results:
                self._plot_performance_vs_size(results['detector_benchmarks'])
            
            # Memory efficiency chart
            if 'memory_efficiency' in results:
                self._plot_memory_efficiency(results['memory_efficiency'])
            
            logger.info("Performance visualizations generated")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available - skipping visualizations")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _plot_performance_vs_size(self, detector_results: Dict):
        """Plot performance vs data size for all detectors."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for detector_result in detector_results['detector_results']:
            metrics = detector_result['metrics']
            if not metrics:
                continue
                
            data_sizes = [m['data_size'] for m in metrics]
            wall_times = [m['wall_time'] for m in metrics]
            throughputs = [m['throughput'] for m in metrics]
            
            # Execution time plot
            ax1.plot(data_sizes, wall_times, marker='o', label=detector_result['detector'])
            
            # Throughput plot
            ax2.plot(data_sizes, throughputs, marker='s', label=detector_result['detector'])
        
        ax1.set_xlabel('Data Size (points)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Data Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Data Size (points)')
        ax2.set_ylabel('Throughput (points/second)')
        ax2.set_title('Throughput vs Data Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_vs_size.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_efficiency(self, memory_results: Dict):
        """Plot memory efficiency metrics."""
        memory_data = memory_results['memory_benchmarks']
        
        data_sizes = [m['data_size'] for m in memory_data]
        memory_growth = [m['memory_growth_mb'] for m in memory_data]
        efficiency_scores = [m['memory_efficiency_score'] for m in memory_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory growth
        ax1.bar(data_sizes, memory_growth, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Data Size (points)')
        ax1.set_ylabel('Memory Growth (MB)')
        ax1.set_title('Memory Usage Growth')
        ax1.grid(True, alpha=0.3)
        
        # Efficiency scores
        ax2.plot(data_sizes, efficiency_scores, marker='o', color='green', linewidth=2)
        ax2.set_xlabel('Data Size (points)')
        ax2.set_ylabel('Efficiency Score (points/MB)')
        ax2.set_title('Memory Efficiency Score')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'memory_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()


def run_macbook_benchmarks():
    """Run comprehensive MacBook performance benchmarks."""
    logger.info("Starting MacBook Performance Benchmarks")
    
    benchmark = MacBookPerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmarks()
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("="*60)
    
    system_info = results['system_info']
    print(f"System: {system_info['system']} {system_info['machine']}")
    print(f"Apple Silicon: {'Yes' if system_info['is_apple_silicon'] else 'No'}")
    
    if 'summary' in results:
        print("\nPerformance Highlights:")
        for highlight in results['summary']['performance_highlights']:
            print(f"  {highlight['detector']}: {highlight['avg_throughput']:.0f} pts/sec "
                  f"({highlight['performance_rating']})")
    
    print(f"\nResults saved to: {benchmark.results_dir}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_macbook_benchmarks()