"""
Machine Learning module for Advanced Pattern Scanner.

This module contains ML-based pattern validation and confidence scoring
components optimized for Apple Silicon performance.
"""

from .model_manager import ModelManager, PatternCNNLSTM
from .pattern_validator import PatternValidator
from .hybrid_validator import HybridValidator
from .synthetic_data_generator import SyntheticDataGenerator
from .train_model import ModelTrainer

__all__ = [
    'ModelManager',
    'PatternCNNLSTM', 
    'PatternValidator',
    'HybridValidator',
    'SyntheticDataGenerator',
    'ModelTrainer'
]