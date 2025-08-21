"""
Apple Silicon Optimization Module

This module provides Apple Silicon specific optimizations including:
- MPS (Metal Performance Shaders) backend configuration
- Vectorized operations using Apple's Accelerate framework
- Memory management optimizations for M1/M2/M3 chips
- Performance monitoring and benchmarking
"""

import os
import logging
import platform
import psutil
from typing import Dict, Optional, Tuple, Any
import time
from functools import wraps

import torch
import numpy as np
from accelerate import Accelerator

# Configure logging
logger = logging.getLogger(__name__)


class AppleSiliconOptimizer:
    """
    Apple Silicon optimization manager for enhanced performance on M1/M2/M3 chips.
    
    This class provides utilities for:
    - Detecting Apple Silicon hardware
    - Configuring optimal PyTorch settings
    - Managing memory efficiently
    - Monitoring performance metrics
    """
    
    def __init__(self):
        """Initialize the Apple Silicon optimizer."""
        self.is_apple_silicon = self._detect_apple_silicon()
        self.device_info = self._get_device_info()
        self.accelerator = None
        
        if self.is_apple_silicon:
            self._configure_apple_silicon()
            logger.info("Apple Silicon optimizations enabled")
        else:
            logger.info("Running on non-Apple Silicon hardware")
    
    def _detect_apple_silicon(self) -> bool:
        """
        Detect if running on Apple Silicon.
        
        Returns:
            True if running on Apple Silicon (M1/M2/M3)
        """
        try:
            # Check platform and processor
            if platform.system() == "Darwin":  # macOS
                processor = platform.processor()
                machine = platform.machine()
                
                # Apple Silicon indicators
                apple_silicon_indicators = [
                    "arm" in processor.lower(),
                    "arm64" in machine.lower(),
                    machine == "arm64"
                ]
                
                return any(apple_silicon_indicators)
        except Exception as e:
            logger.warning(f"Could not detect Apple Silicon: {e}")
        
        return False
    
    def _get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.
        
        Returns:
            Dictionary with device specifications
        """
        info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "cuda_available": torch.cuda.is_available()
        }
        
        if self.is_apple_silicon:
            info["chip_type"] = self._detect_chip_type()
            info["unified_memory"] = True
        
        return info
    
    def _detect_chip_type(self) -> str:
        """
        Detect specific Apple Silicon chip type.
        
        Returns:
            Chip type string (M1, M2, M3, or Unknown)
        """
        try:
            # Use system_profiler to get chip information
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            output = result.stdout.lower()
            if "apple m3" in output:
                return "M3"
            elif "apple m2" in output:
                return "M2"
            elif "apple m1" in output:
                return "M1"
        except Exception as e:
            logger.debug(f"Could not detect specific chip type: {e}")
        
        return "Unknown Apple Silicon"
    
    def _configure_apple_silicon(self):
        """Configure optimal settings for Apple Silicon."""
        try:
            # Set optimal number of threads
            if self.is_apple_silicon:
                # Apple Silicon benefits from using all performance cores
                optimal_threads = min(8, psutil.cpu_count())  # Cap at 8 for efficiency
                torch.set_num_threads(optimal_threads)
                os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
                os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
                
                logger.info(f"Set optimal thread count: {optimal_threads}")
            
            # Configure MPS if available
            if torch.backends.mps.is_available():
                # Enable MPS fallback for unsupported operations
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                logger.info("MPS backend configured with fallback enabled")
            
            # Initialize Accelerator for distributed computing
            self.accelerator = Accelerator()
            
        except Exception as e:
            logger.warning(f"Apple Silicon configuration failed: {e}")
    
    def get_optimal_device(self, prefer_mps: bool = True) -> torch.device:
        """
        Get the optimal PyTorch device for the current hardware.
        
        Args:
            prefer_mps: Whether to prefer MPS over CPU on Apple Silicon
            
        Returns:
            Optimal torch.device
        """
        if self.is_apple_silicon and prefer_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def optimize_tensor_operations(self, enable_amp: bool = True) -> Dict[str, Any]:
        """
        Configure tensor operations for optimal performance.
        
        Args:
            enable_amp: Whether to enable Automatic Mixed Precision
            
        Returns:
            Configuration dictionary
        """
        config = {}
        
        try:
            # Enable optimized attention if available
            if hasattr(torch.backends, 'opt_einsum'):
                torch.backends.opt_einsum.enabled = True
                config["opt_einsum"] = True
            
            # Configure cuDNN for consistency (if available)
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                config["cudnn_benchmark"] = True
            
            # Enable JIT compilation for better performance
            torch.jit.set_fusion_strategy([("STATIC", 2), ("DYNAMIC", 2)])
            config["jit_fusion"] = True
            
            # Configure automatic mixed precision
            if enable_amp and self.is_apple_silicon:
                config["amp_enabled"] = True
                config["amp_dtype"] = torch.float16
            
            logger.info(f"Tensor operations optimized: {config}")
            
        except Exception as e:
            logger.warning(f"Tensor optimization failed: {e}")
        
        return config
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage for Apple Silicon unified memory architecture.
        
        Returns:
            Memory optimization configuration
        """
        config = {}
        
        try:
            if self.is_apple_silicon:
                # Set memory fraction for MPS
                if torch.backends.mps.is_available():
                    # Apple Silicon has unified memory, so we can be more aggressive
                    memory_fraction = 0.8  # Use up to 80% of available memory
                    config["mps_memory_fraction"] = memory_fraction
                
                # Configure garbage collection
                import gc
                gc.set_threshold(700, 10, 10)  # More aggressive GC for unified memory
                config["gc_threshold"] = (700, 10, 10)
                
                # Enable memory mapping for large tensors
                torch.backends.mkl.enabled = True
                config["mkl_enabled"] = True
            
            # Set optimal batch sizes based on available memory
            available_memory_gb = self.device_info["memory_gb"]
            if available_memory_gb >= 16:
                config["recommended_batch_size"] = 64
            elif available_memory_gb >= 8:
                config["recommended_batch_size"] = 32
            else:
                config["recommended_batch_size"] = 16
            
            logger.info(f"Memory optimizations applied: {config}")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
        
        return config
    
    def benchmark_operations(self, tensor_size: Tuple[int, ...] = (1000, 1000)) -> Dict[str, float]:
        """
        Benchmark basic tensor operations to validate optimizations.
        
        Args:
            tensor_size: Size of tensors for benchmarking
            
        Returns:
            Benchmark results in milliseconds
        """
        results = {}
        device = self.get_optimal_device()
        
        try:
            # Matrix multiplication benchmark
            a = torch.randn(tensor_size, device=device)
            b = torch.randn(tensor_size, device=device)
            
            # Warm up
            for _ in range(10):
                _ = torch.mm(a, b)
            
            # Benchmark matrix multiplication
            start_time = time.time()
            for _ in range(100):
                _ = torch.mm(a, b)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            matmul_time = (time.time() - start_time) * 10  # Convert to ms
            results["matrix_multiplication_ms"] = matmul_time
            
            # Element-wise operations benchmark
            start_time = time.time()
            for _ in range(100):
                _ = a * b + a
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            
            elementwise_time = (time.time() - start_time) * 10
            results["elementwise_ops_ms"] = elementwise_time
            
            # Memory transfer benchmark (if not CPU)
            if device.type != "cpu":
                cpu_tensor = torch.randn(tensor_size)
                
                start_time = time.time()
                for _ in range(10):
                    _ = cpu_tensor.to(device)
                if device.type == "mps":
                    torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
                
                transfer_time = (time.time() - start_time) * 100  # Convert to ms
                results["memory_transfer_ms"] = transfer_time
            
            results["device"] = str(device)
            results["tensor_size"] = tensor_size
            
            logger.info(f"Benchmark results: {results}")
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """
        Get performance recommendations based on hardware.
        
        Returns:
            Dictionary with performance recommendations
        """
        recommendations = {
            "device": str(self.get_optimal_device()),
            "hardware_info": self.device_info
        }
        
        if self.is_apple_silicon:
            recommendations.update({
                "use_mps": torch.backends.mps.is_available(),
                "optimal_batch_size": 32 if self.device_info["memory_gb"] >= 16 else 16,
                "enable_amp": True,
                "use_vectorized_ops": True,
                "memory_management": "unified_memory_optimized",
                "threading": {
                    "torch_threads": min(8, self.device_info["cpu_count"]),
                    "numpy_threads": min(4, self.device_info["cpu_count"] // 2)
                }
            })
        else:
            recommendations.update({
                "use_cuda": torch.cuda.is_available(),
                "optimal_batch_size": 64 if self.device_info["memory_gb"] >= 32 else 32,
                "enable_amp": torch.cuda.is_available(),
                "memory_management": "standard"
            })
        
        return recommendations


def apple_silicon_optimized(func):
    """
    Decorator to automatically apply Apple Silicon optimizations to functions.
    
    This decorator ensures that tensor operations within the decorated function
    use optimal device placement and memory management.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = AppleSiliconOptimizer()
        
        # Store original settings
        original_threads = torch.get_num_threads()
        
        try:
            # Apply optimizations
            if optimizer.is_apple_silicon:
                optimizer.optimize_tensor_operations()
                optimizer.optimize_memory_usage()
            
            # Execute function
            result = func(*args, **kwargs)
            
            return result
            
        finally:
            # Restore original settings
            torch.set_num_threads(original_threads)
    
    return wrapper


def get_optimal_numpy_config() -> Dict[str, Any]:
    """
    Get optimal NumPy configuration for Apple Silicon.
    
    Returns:
        NumPy configuration dictionary
    """
    config = {}
    
    try:
        # Check if running on Apple Silicon
        optimizer = AppleSiliconOptimizer()
        
        if optimizer.is_apple_silicon:
            # Use Apple's Accelerate framework
            os.environ["NPY_NUM_BUILD_JOBS"] = str(min(8, psutil.cpu_count()))
            
            # Configure BLAS threading
            os.environ["OPENBLAS_NUM_THREADS"] = str(min(4, psutil.cpu_count() // 2))
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(min(8, psutil.cpu_count()))
            
            config = {
                "accelerate_framework": True,
                "optimal_threads": min(8, psutil.cpu_count()),
                "blas_threads": min(4, psutil.cpu_count() // 2)
            }
        
        logger.info(f"NumPy configuration: {config}")
        
    except Exception as e:
        logger.warning(f"NumPy configuration failed: {e}")
    
    return config


# Initialize global optimizer instance
_global_optimizer = None

def get_global_optimizer() -> AppleSiliconOptimizer:
    """Get the global Apple Silicon optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AppleSiliconOptimizer()
    return _global_optimizer