"""
Core data models for the Advanced Pattern Scanner.

This module defines the fundamental data structures used throughout the system
for pattern detection, configuration, and analysis.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class Pattern:
    """
    Represents a detected stock pattern with all relevant information.
    
    This is the core data structure that holds pattern detection results,
    trading information, and metadata for analysis and visualization.
    """
    
    # Basic pattern information
    type: str  # Pattern type (e.g., "Head and Shoulders", "Double Bottom")
    symbol: str  # Stock symbol
    timeframe: str  # Timeframe (e.g., "1d", "1h")
    
    # Key points defining the pattern
    key_points: List[Tuple[int, float]]  # (index, price) pairs for pattern structure
    
    # Pattern confidence and scoring
    confidence: float  # ML confidence score (0-1)
    traditional_score: float  # Rule-based score (0-1)
    combined_score: float  # Final combined score (0-1)
    
    # Trading information
    entry_price: float  # Recommended entry price
    target_price: float  # Price target based on pattern
    stop_loss: float  # Stop loss level
    risk_reward_ratio: float  # Risk/reward ratio
    
    # Pattern timeline
    formation_start: datetime  # When pattern formation began
    formation_end: datetime  # When pattern formation completed
    breakout_date: Optional[datetime] = None  # When breakout occurred (if any)
    
    # Pattern status and validation
    status: str = "forming"  # "forming", "confirmed", "failed"
    
    # Volume analysis
    volume_confirmation: bool = False  # Whether volume confirms the pattern
    avg_volume_ratio: float = 1.0  # Volume ratio vs average
    
    # Pattern characteristics
    pattern_height: float = 0.0  # Height of the pattern (price range)
    duration_days: int = 0  # Duration of pattern formation in days
    
    # Detection metadata
    detection_method: str = "hybrid"  # "traditional", "ml", "hybrid"
    detection_timestamp: datetime = None  # When pattern was detected
    
    def __post_init__(self):
        """Initialize computed fields after object creation."""
        if self.detection_timestamp is None:
            self.detection_timestamp = datetime.now()
    
    @property
    def is_bullish(self) -> bool:
        """Determine if pattern is bullish based on type and structure."""
        bullish_patterns = {
            "Double Bottom", "Cup and Handle", "Ascending Triangle",
            "Inverse Head and Shoulders", "Bull Flag", "Bull Pennant"
        }
        return self.type in bullish_patterns
    
    @property
    def is_bearish(self) -> bool:
        """Determine if pattern is bearish based on type and structure."""
        bearish_patterns = {
            "Double Top", "Head and Shoulders", "Descending Triangle",
            "Bear Flag", "Bear Pennant"
        }
        return self.type in bearish_patterns
    
    @property
    def potential_profit_percent(self) -> float:
        """Calculate potential profit percentage from entry to target."""
        if self.entry_price == 0:
            return 0.0
        return ((self.target_price - self.entry_price) / self.entry_price) * 100
    
    @property
    def risk_percent(self) -> float:
        """Calculate risk percentage from entry to stop loss."""
        if self.entry_price == 0:
            return 0.0
        return abs((self.stop_loss - self.entry_price) / self.entry_price) * 100


@dataclass
class PatternConfig:
    """
    Configuration settings for pattern detection and validation.
    
    This class holds all configurable parameters that control how patterns
    are detected, validated, and scored throughout the system.
    """
    
    # Detection thresholds
    min_confidence: float = 0.7  # Minimum confidence score to report pattern
    min_traditional_score: float = 0.6  # Minimum traditional algorithm score
    min_combined_score: float = 0.65  # Minimum combined score
    
    # Pattern duration constraints
    min_pattern_duration: int = 20  # Minimum days for pattern formation
    max_pattern_duration: int = 200  # Maximum days for pattern formation
    
    # Volume analysis settings
    volume_confirmation_required: bool = True  # Require volume confirmation
    min_volume_ratio: float = 1.2  # Minimum volume ratio for confirmation
    volume_lookback_days: int = 20  # Days to look back for average volume
    
    # Pattern-specific tolerances
    head_shoulders_tolerance: float = 0.05  # 5% tolerance for H&S symmetry
    double_pattern_tolerance: float = 0.03  # 3% tolerance for double patterns
    cup_handle_depth_min: float = 0.15  # Minimum cup depth (15%)
    cup_handle_depth_max: float = 0.50  # Maximum cup depth (50%)
    triangle_convergence_min: float = 0.02  # Minimum convergence for triangles
    
    # Technical indicator settings
    rsi_period: int = 14  # RSI calculation period
    macd_fast: int = 12  # MACD fast EMA period
    macd_slow: int = 26  # MACD slow EMA period
    macd_signal: int = 9  # MACD signal line period
    
    # ML model settings
    model_path: str = "models/pattern_classifier.pth"  # Path to trained model
    use_gpu: bool = True  # Use GPU acceleration if available
    batch_size: int = 32  # Batch size for ML inference
    sequence_length: int = 60  # Input sequence length for ML model
    
    # Data processing settings
    data_cache_ttl: int = 3600  # Cache TTL in seconds (1 hour)
    max_cache_size: int = 1000  # Maximum number of cached symbols
    data_validation_strict: bool = True  # Strict data validation
    
    # Performance settings
    max_concurrent_requests: int = 10  # Max concurrent API requests
    request_timeout: int = 30  # API request timeout in seconds
    enable_parallel_processing: bool = True  # Enable parallel pattern detection
    
    # Risk management
    max_risk_percent: float = 2.0  # Maximum risk per trade (%)
    min_risk_reward_ratio: float = 1.5  # Minimum risk/reward ratio
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters and return list of errors.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Validate confidence thresholds
        if not 0 <= self.min_confidence <= 1:
            errors.append("min_confidence must be between 0 and 1")
        if not 0 <= self.min_traditional_score <= 1:
            errors.append("min_traditional_score must be between 0 and 1")
        if not 0 <= self.min_combined_score <= 1:
            errors.append("min_combined_score must be between 0 and 1")
        
        # Validate duration constraints
        if self.min_pattern_duration <= 0:
            errors.append("min_pattern_duration must be positive")
        if self.max_pattern_duration <= self.min_pattern_duration:
            errors.append("max_pattern_duration must be greater than min_pattern_duration")
        
        # Validate volume settings
        if self.min_volume_ratio <= 0:
            errors.append("min_volume_ratio must be positive")
        if self.volume_lookback_days <= 0:
            errors.append("volume_lookback_days must be positive")
        
        # Validate tolerances
        if not 0 <= self.head_shoulders_tolerance <= 1:
            errors.append("head_shoulders_tolerance must be between 0 and 1")
        if not 0 <= self.double_pattern_tolerance <= 1:
            errors.append("double_pattern_tolerance must be between 0 and 1")
        
        # Validate cup and handle settings
        if not 0 < self.cup_handle_depth_min < self.cup_handle_depth_max < 1:
            errors.append("cup_handle_depth values must be: 0 < min < max < 1")
        
        # Validate risk management
        if self.max_risk_percent <= 0:
            errors.append("max_risk_percent must be positive")
        if self.min_risk_reward_ratio <= 0:
            errors.append("min_risk_reward_ratio must be positive")
        
        return errors
    
    @classmethod
    def create_conservative(cls) -> 'PatternConfig':
        """Create a conservative configuration with higher thresholds."""
        return cls(
            min_confidence=0.8,
            min_traditional_score=0.75,
            min_combined_score=0.75,
            volume_confirmation_required=True,
            min_volume_ratio=1.5,
            min_risk_reward_ratio=2.0
        )
    
    @classmethod
    def create_aggressive(cls) -> 'PatternConfig':
        """Create an aggressive configuration with lower thresholds."""
        return cls(
            min_confidence=0.6,
            min_traditional_score=0.5,
            min_combined_score=0.55,
            volume_confirmation_required=False,
            min_volume_ratio=1.0,
            min_risk_reward_ratio=1.0
        )


@dataclass
class MarketData:
    """
    Container for market data used in pattern detection.
    
    This class holds the processed market data along with computed
    technical indicators needed for pattern analysis.
    """
    
    symbol: str
    timeframe: str
    data: np.ndarray  # OHLCV data as numpy array for performance
    timestamps: List[datetime]
    
    # Technical indicators (computed lazily)
    _rsi: Optional[np.ndarray] = None
    _macd: Optional[Dict[str, np.ndarray]] = None
    _volume_sma: Optional[np.ndarray] = None
    _price_sma_20: Optional[np.ndarray] = None
    _price_sma_50: Optional[np.ndarray] = None
    
    @property
    def open_prices(self) -> np.ndarray:
        """Get open prices."""
        return self.data[:, 0]
    
    @property
    def high_prices(self) -> np.ndarray:
        """Get high prices."""
        return self.data[:, 1]
    
    @property
    def low_prices(self) -> np.ndarray:
        """Get low prices."""
        return self.data[:, 2]
    
    @property
    def close_prices(self) -> np.ndarray:
        """Get close prices."""
        return self.data[:, 3]
    
    @property
    def volumes(self) -> np.ndarray:
        """Get volumes."""
        return self.data[:, 4]
    
    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.data)