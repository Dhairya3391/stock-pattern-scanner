# ML Validation System

This module provides machine learning-based validation for stock pattern detection, optimized for Apple Silicon (MacBook M1/M2/M3) performance.

## Overview

The ML validation system combines traditional rule-based pattern detection with modern machine learning to create a hybrid approach that reduces false positives and improves pattern detection accuracy.

## Components

### 1. ModelManager (`model_manager.py`)
- Manages PyTorch CNN-LSTM models optimized for Apple Silicon MPS backend
- Handles model loading, inference, and batch prediction
- Provides fallback to Random Forest models
- Supports both pre-trained and synthetic models for demonstration

**Key Features:**
- Apple Silicon optimization with MPS backend
- Automatic device selection (MPS > CUDA > CPU)
- Graceful fallback between model types
- Batch processing capabilities

### 2. PatternValidator (`pattern_validator.py`)
- Validates patterns detected by traditional algorithms using ML models
- Combines ML confidence with technical indicator analysis
- Provides detailed validation reports with component scores

**Validation Components:**
- ML model confidence (40% weight)
- Traditional algorithm score (30% weight)
- Volume confirmation (20% weight)
- Technical indicators (10% weight)

### 3. HybridValidator (`hybrid_validator.py`)
- Main validation orchestrator combining traditional and ML approaches
- Implements sophisticated decision logic with pattern-specific thresholds
- Provides comprehensive validation statistics and reporting

**Decision Logic:**
- Primary: Traditional algorithm detection
- Secondary: ML model validation
- Tertiary: Technical and volume confirmation
- Final: Hybrid scoring with pattern-specific adjustments

### 4. SyntheticDataGenerator (`synthetic_data_generator.py`)
- Generates synthetic market data with embedded patterns
- Based on reference algorithm specifications (HnS.md, doubleBottom.md, CupHandle.md)
- Creates realistic OHLCV data with proper volume patterns

**Supported Patterns:**
- Head and Shoulders
- Double Bottom/Top
- Cup and Handle
- Ascending/Descending Triangles
- Random Walk (No Pattern)

### 5. ModelTrainer (`train_model.py`)
- Complete training pipeline for both CNN-LSTM and Random Forest models
- Handles data generation, model training, validation, and saving
- Includes early stopping, learning rate scheduling, and performance monitoring

**Training Features:**
- Synthetic data generation
- Train/validation splits with stratification
- Early stopping with patience
- Model checkpointing
- Performance evaluation

## Model Architecture

### CNN-LSTM Hybrid Model
```
Input: (batch_size, sequence_length=60, features=10)
├── 1D CNN Layers (local pattern extraction)
│   ├── Conv1d(10→64, kernel=3) + ReLU + BatchNorm + Dropout
│   ├── Conv1d(64→128, kernel=3) + ReLU + BatchNorm + Dropout
│   └── Conv1d(128→256, kernel=3) + ReLU + BatchNorm + MaxPool + Dropout
├── Bidirectional LSTM (temporal dependencies)
│   └── LSTM(256→256, 2 layers, bidirectional, dropout=0.2)
├── Multi-head Attention (8 heads)
├── Classification Head (pattern type)
│   └── Linear(512→256→128→6) + ReLU + Dropout
└── Confidence Head (confidence score)
    └── Linear(512→128→64→1) + Sigmoid
```

### Random Forest Fallback
- 200 estimators, max_depth=15
- Feature selection and scaling
- Robust to missing data
- Fast inference for real-time use

## Usage

### Basic Usage
```python
from advanced_pattern_scanner.core.models import PatternConfig
from advanced_pattern_scanner.ml import HybridValidator

# Initialize
config = PatternConfig()
validator = HybridValidator(config)

# Validate a pattern
is_valid, confidence, details = validator.validate_pattern_hybrid(pattern, market_data)
```

### Training Models
```python
from advanced_pattern_scanner.ml import ModelTrainer

trainer = ModelTrainer(config)
results = trainer.train_all_models(num_samples=5000)
```

### Batch Validation
```python
results = validator.batch_validate_hybrid(patterns, market_data_list)
stats = validator.get_validation_statistics(results)
```

## Configuration

Key configuration parameters in `PatternConfig`:

```python
# ML Model Settings
model_path: str = "models/pattern_classifier.pth"
use_gpu: bool = True
batch_size: int = 32
sequence_length: int = 60

# Validation Thresholds
min_confidence: float = 0.7
min_traditional_score: float = 0.6
min_combined_score: float = 0.65

# Pattern-Specific Settings
head_shoulders_tolerance: float = 0.05
double_pattern_tolerance: float = 0.03
cup_handle_depth_min: float = 0.15
```

## Performance Optimization

### Apple Silicon Optimizations
- **MPS Backend**: Utilizes Metal Performance Shaders for GPU acceleration
- **Memory Management**: Efficient memory usage with streaming for large datasets
- **Vectorized Operations**: ARM64 optimized NumPy operations
- **Parallel Processing**: Multi-threading for independent pattern detection

### Model Optimizations
- **Lightweight Architecture**: Optimized for real-time inference
- **Batch Processing**: Efficient batch prediction capabilities
- **Model Compression**: Quantization-ready architecture
- **Caching**: Intelligent feature caching for repeated patterns

## Testing

Run the test suite:
```bash
python3 advanced_pattern_scanner/ml/test_ml_system.py
```

Run training demo:
```bash
python3 advanced_pattern_scanner/ml/demo_training.py
```

## Requirements

### Core Dependencies
- PyTorch (with MPS support for Apple Silicon)
- NumPy
- Scikit-learn
- Pandas
- SciPy

### Optional Dependencies
- Core ML (for additional Apple Silicon optimization)
- Hugging Face Transformers (for pre-trained models)

## Model Files

Trained models are saved in the `models/` directory:
- `pattern_cnn_lstm.pth`: CNN-LSTM model checkpoint
- `pattern_random_forest.joblib`: Random Forest model
- `feature_scaler.joblib`: Feature scaler for Random Forest
- `feature_importance.npy`: Feature importance scores

## Pattern-Specific Thresholds

Different patterns have different validation thresholds:

| Pattern | Traditional | ML | Combined |
|---------|-------------|----| ---------|
| Head & Shoulders | 0.70 | 0.75 | 0.72 |
| Double Bottom/Top | 0.65 | 0.70 | 0.68 |
| Cup & Handle | 0.75 | 0.80 | 0.77 |
| Triangles | 0.60 | 0.65 | 0.62 |

## Error Handling

The system includes comprehensive error handling:
- **Graceful Degradation**: Falls back to traditional validation if ML fails
- **Model Loading**: Creates synthetic models if trained models unavailable
- **Data Validation**: Handles missing or invalid data gracefully
- **Memory Management**: Prevents memory leaks during batch processing

## Logging

Comprehensive logging at multiple levels:
- **INFO**: General operation status
- **WARNING**: Non-critical issues (model fallbacks, data issues)
- **ERROR**: Critical failures requiring attention
- **DEBUG**: Detailed validation information

## Future Enhancements

Planned improvements:
1. **Core ML Integration**: Native Apple Silicon optimization
2. **Online Learning**: Adaptive model updates based on market feedback
3. **Ensemble Methods**: Multiple model voting systems
4. **Pattern Clustering**: Unsupervised pattern discovery
5. **Real-time Streaming**: Live market data processing
6. **Model Interpretability**: SHAP/LIME explanations for predictions