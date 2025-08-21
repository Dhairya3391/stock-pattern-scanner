# Implementation Plan

- [x] 1. Create new modular project structure and core foundation
  - Create new `advanced_pattern_scanner/` directory with modular structure: `core/`, `patterns/`, `ml/`, `ui/`, `tests/`
  - Implement core data models (Pattern, PatternConfig classes) in `core/models.py`
  - Set up optimized dependencies for MacBook in `requirements.txt` (PyTorch MPS, Core ML, scipy, yfinance)
  - Create `core/data_manager.py` with efficient data fetching, caching, and preprocessing
  - _Requirements: 5.1, 5.4, 1.1, 1.3_

- [x] 2. Implement reference-based pattern detection algorithms
  - Create `patterns/head_shoulders.py` implementing exact HnS.md algorithm (findPeaks, identifyHS, drawNeckline, confirmNecklineBreak, analyzeVolume, calculatePriceTarget)
  - Create `patterns/double_bottom.py` implementing doubleBottom.md methodology with scipy.signal.find_peaks and 2% tolerance validation
  - Create `patterns/cup_handle.py` implementing CupHandle.md algorithm with zigzag indicators, U-shape validation, and duration constraints
  - Test each pattern detector using exact examples from reference documents (Stock ABC/XYZ for double bottom)
  - _Requirements: 2.1, 2.2, 2.3, 6.1_

- [x] 3. Build ML validation system optimized for MacBook
  - Create `ml/model_manager.py` with PyTorch MPS backend for Apple Silicon optimization
  - Implement `ml/pattern_validator.py` that validates patterns detected by reference algorithms
  - Train lightweight CNN-LSTM model using synthetic data generated from reference algorithm specifications
  - Create hybrid validation system that uses reference algorithms as primary detection with ML confirmation
  - _Requirements: 3.1, 3.2, 3.4, 4.1, 4.4_

- [x] 4. Create main detection engine and user interface
  - Build `core/pattern_engine.py` that orchestrates all pattern detectors and ML validation
  - Implement `core/pattern_scorer.py` for target price calculation and risk assessment using reference methodologies
  - Create `ui/streamlit_app.py` with interactive Plotly charts showing detected patterns with confidence scores
  - Add batch processing for multiple stock scanning with Apple Silicon optimization
  - _Requirements: 6.1, 6.2, 6.3, 7.1, 7.2, 7.3_

- [x] 5. Clean up legacy code and optimize performance
  - Remove unnecessary code from existing `Improved Stock Scanner.py` and ignore the `stock_scanner.py` for backup.
  - Implement Apple Silicon specific optimizations (MPS, vectorized operations, memory management)
  - Add comprehensive error handling and logging throughout the new modular system
  - Create documentation and examples showing how to use the new reference-based detection system
  - _Requirements: 1.1, 1.2, 1.3, 3.3, 5.4_

- [x] 6. Final validation and testing
  - Test all pattern detectors using exact examples from reference documents (Stock ABC/XYZ for double bottom)
  - Validate that HnS.md, doubleBottom.md, and CupHandle.md algorithms produce expected results
  - Create performance benchmarks for MacBook M1/M2/M3 showing improved speed and accuracy
  - Document accuracy metrics proving the new system matches reference specifications exactly
  - _Requirements: 2.1, 2.2, 2.3, 3.4, 6.3_