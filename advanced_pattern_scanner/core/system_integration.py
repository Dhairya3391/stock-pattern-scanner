"""
System Integration Module

This module provides comprehensive integration of all system components
including Apple Silicon optimizations, error handling, and performance
monitoring. It serves as the main entry point for system initialization.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .apple_silicon_optimizer import get_global_optimizer
from .error_handler import setup_global_error_handling, get_global_logging_manager
from .models import PatternConfig
from .pattern_engine import PatternEngine


class SystemIntegrator:
    """
    Main system integrator that coordinates all components.
    
    This class handles:
    - System initialization and configuration
    - Apple Silicon optimization setup
    - Error handling and logging configuration
    - Performance monitoring setup
    - Component coordination
    """
    
    def __init__(self, config: Optional[PatternConfig] = None):
        """
        Initialize the system integrator.
        
        Args:
            config: Optional pattern configuration (uses defaults if None)
        """
        self.config = config or PatternConfig()
        self.optimizer = None
        self.logging_manager = None
        self.pattern_engine = None
        self.is_initialized = False
        
        # System status
        self.system_status = {
            "initialized": False,
            "apple_silicon_optimized": False,
            "error_handling_setup": False,
            "logging_configured": False,
            "pattern_engine_ready": False
        }
    
    def initialize_system(self) -> Dict[str, Any]:
        """
        Initialize the complete system with all optimizations.
        
        Returns:
            Dictionary with initialization results and system status
        """
        if self.is_initialized:
            return self._get_system_status()
        
        initialization_results = {}
        
        try:
            # 1. Setup global error handling and logging
            setup_global_error_handling()
            self.logging_manager = get_global_logging_manager()
            self.system_status["error_handling_setup"] = True
            self.system_status["logging_configured"] = True
            initialization_results["logging"] = "✅ Configured"
            
            # 2. Initialize Apple Silicon optimizer
            self.optimizer = get_global_optimizer()
            
            if self.optimizer.is_apple_silicon:
                # Apply optimizations
                tensor_config = self.optimizer.optimize_tensor_operations(enable_amp=True)
                memory_config = self.optimizer.optimize_memory_usage()
                
                self.system_status["apple_silicon_optimized"] = True
                initialization_results["apple_silicon"] = {
                    "status": "✅ Optimized",
                    "chip_type": self.optimizer.device_info.get("chip_type", "Unknown"),
                    "device": str(self.optimizer.get_optimal_device()),
                    "tensor_config": tensor_config,
                    "memory_config": memory_config
                }
            else:
                initialization_results["apple_silicon"] = {
                    "status": "ℹ️  Not Apple Silicon",
                    "device": str(self.optimizer.get_optimal_device())
                }
            
            # 3. Initialize pattern engine with optimized configuration
            optimized_config = self._create_optimized_config()
            self.pattern_engine = PatternEngine(optimized_config)
            self.system_status["pattern_engine_ready"] = True
            initialization_results["pattern_engine"] = "✅ Ready"
            
            # 4. Validate system components
            validation_results = self._validate_system_components()
            initialization_results["validation"] = validation_results
            
            # 5. Run system benchmark
            if self.optimizer.is_apple_silicon:
                benchmark_results = self.optimizer.benchmark_operations()
                initialization_results["benchmark"] = benchmark_results
            
            self.is_initialized = True
            self.system_status["initialized"] = True
            
            # Log successful initialization
            self.logging_manager.log_performance(
                "system_initialization", 
                0.0,  # Will be measured by decorator
                True,
                {"components": len(initialization_results)}
            )
            
        except Exception as e:
            error_msg = f"System initialization failed: {e}"
            initialization_results["error"] = error_msg
            
            if self.logging_manager:
                from .error_handler import PatternScannerError, ErrorCategory, ErrorSeverity
                error = PatternScannerError(
                    error_msg,
                    ErrorCategory.SYSTEM_ERROR,
                    ErrorSeverity.CRITICAL
                )
                self.logging_manager.log_error(error)
        
        return initialization_results
    
    def _create_optimized_config(self) -> PatternConfig:
        """
        Create optimized configuration based on hardware capabilities.
        
        Returns:
            Optimized PatternConfig instance
        """
        # Start with base configuration
        optimized_config = PatternConfig(
            # Copy existing settings
            min_confidence=self.config.min_confidence,
            min_pattern_duration=self.config.min_pattern_duration,
            max_pattern_duration=self.config.max_pattern_duration,
            volume_confirmation_required=self.config.volume_confirmation_required,
            min_volume_ratio=self.config.min_volume_ratio
        )
        
        if self.optimizer and self.optimizer.is_apple_silicon:
            # Apply Apple Silicon optimizations
            recommendations = self.optimizer.get_performance_recommendations()
            
            # Update configuration with optimized settings
            optimized_config.use_gpu = True
            optimized_config.enable_parallel_processing = True
            optimized_config.batch_size = recommendations.get("optimal_batch_size", 32)
            optimized_config.max_concurrent_requests = min(4, os.cpu_count() or 4)
            
            # Memory optimizations
            memory_gb = self.optimizer.device_info.get("memory_gb", 8)
            if memory_gb >= 16:
                optimized_config.data_cache_ttl = 7200  # 2 hours for high memory systems
            else:
                optimized_config.data_cache_ttl = 3600  # 1 hour for lower memory systems
        
        return optimized_config
    
    def _validate_system_components(self) -> Dict[str, str]:
        """
        Validate that all system components are working correctly.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        try:
            # Validate optimizer
            if self.optimizer:
                device = self.optimizer.get_optimal_device()
                validation_results["optimizer"] = f"✅ Device: {device}"
            else:
                validation_results["optimizer"] = "❌ Not initialized"
            
            # Validate logging
            if self.logging_manager:
                validation_results["logging"] = "✅ Active"
            else:
                validation_results["logging"] = "❌ Not configured"
            
            # Validate pattern engine
            if self.pattern_engine:
                stats = self.pattern_engine.get_engine_stats()
                validation_results["pattern_engine"] = "✅ Ready"
            else:
                validation_results["pattern_engine"] = "❌ Not initialized"
            
            # Validate data cache
            cache_dir = Path(".cache/market_data")
            if cache_dir.exists():
                validation_results["cache"] = "✅ Available"
            else:
                validation_results["cache"] = "ℹ️  Will be created on first use"
            
        except Exception as e:
            validation_results["error"] = f"❌ Validation failed: {e}"
        
        return validation_results
    
    def get_pattern_engine(self) -> Optional[PatternEngine]:
        """
        Get the initialized pattern engine.
        
        Returns:
            PatternEngine instance or None if not initialized
        """
        if not self.is_initialized:
            self.initialize_system()
        
        return self.pattern_engine
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "system_status": self.system_status,
            "is_initialized": self.is_initialized
        }
        
        if self.optimizer:
            info["hardware"] = self.optimizer.device_info
            info["performance_recommendations"] = self.optimizer.get_performance_recommendations()
        
        if self.pattern_engine:
            info["engine_stats"] = self.pattern_engine.get_engine_stats()
        
        if self.logging_manager:
            info["error_summary"] = self.logging_manager.get_error_summary()
            info["performance_summary"] = self.logging_manager.get_performance_summary()
        
        return info
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "status": "✅ Already initialized",
            "system_info": self.get_system_info()
        }
    
    def shutdown_system(self):
        """Gracefully shutdown the system."""
        if self.pattern_engine:
            # Clear caches
            self.pattern_engine.clear_cache()
        
        if self.logging_manager:
            # Log shutdown
            self.logging_manager.log_performance("system_shutdown", 0.0, True)
        
        self.is_initialized = False
        self.system_status["initialized"] = False


# Global system integrator instance
_global_integrator: Optional[SystemIntegrator] = None


def get_global_integrator(config: Optional[PatternConfig] = None) -> SystemIntegrator:
    """
    Get the global system integrator instance.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        SystemIntegrator instance
    """
    global _global_integrator
    
    if _global_integrator is None:
        _global_integrator = SystemIntegrator(config)
    
    return _global_integrator


def initialize_advanced_pattern_scanner(config: Optional[PatternConfig] = None) -> Dict[str, Any]:
    """
    Initialize the complete Advanced Pattern Scanner system.
    
    This is the main entry point for system initialization. It sets up
    all components with optimal configuration for the current hardware.
    
    Args:
        config: Optional pattern configuration
        
    Returns:
        Dictionary with initialization results
    """
    integrator = get_global_integrator(config)
    return integrator.initialize_system()


def get_optimized_pattern_engine(config: Optional[PatternConfig] = None) -> PatternEngine:
    """
    Get an optimized pattern engine instance.
    
    This function ensures the system is properly initialized and returns
    a pattern engine with all optimizations applied.
    
    Args:
        config: Optional pattern configuration
        
    Returns:
        Optimized PatternEngine instance
    """
    integrator = get_global_integrator(config)
    engine = integrator.get_pattern_engine()
    
    if engine is None:
        raise RuntimeError("Failed to initialize pattern engine")
    
    return engine


def get_system_status() -> Dict[str, Any]:
    """
    Get current system status and information.
    
    Returns:
        Dictionary with comprehensive system status
    """
    integrator = get_global_integrator()
    return integrator.get_system_info()


def benchmark_system_performance() -> Dict[str, Any]:
    """
    Run comprehensive system performance benchmark.
    
    Returns:
        Dictionary with benchmark results
    """
    integrator = get_global_integrator()
    
    if not integrator.is_initialized:
        integrator.initialize_system()
    
    results = {}
    
    # Hardware benchmark
    if integrator.optimizer and integrator.optimizer.is_apple_silicon:
        results["hardware_benchmark"] = integrator.optimizer.benchmark_operations()
    
    # Engine performance
    if integrator.pattern_engine:
        results["engine_stats"] = integrator.pattern_engine.get_engine_stats()
    
    # System info
    results["system_info"] = integrator.get_system_info()
    
    return results