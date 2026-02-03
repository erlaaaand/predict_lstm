"""
Configuration settings for stock prediction system.
Inspired by Renaissance Technologies approach.
"""

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class SystemConfig:
    """System configuration."""
    
    # Application
    APP_TITLE: str = "Quantitative Stock Analysis System"
    APP_ICON: str = "📊"
    LAYOUT: str = "wide"
    
    # Data Collection
    ENABLE_FUNDAMENTAL: bool = True
    ENABLE_TECHNICAL: bool = True
    ENABLE_ALTERNATIVE: bool = True
    ENABLE_SENTIMENT: bool = True
    
    # Data Sources
    MIN_DATA_DAYS: int = 365
    MAX_DATA_DAYS: int = 1825  # 5 years
    DEFAULT_DATA_DAYS: int = 730  # 2 years
    
    # Model Configuration
    MIN_LOOKBACK: int = 30
    MAX_LOOKBACK: int = 120
    DEFAULT_LOOKBACK: int = 60
    
    # Regularization (prevent overfitting)
    DROPOUT_MIN: float = 0.2
    DROPOUT_MAX: float = 0.5
    DEFAULT_DROPOUT: float = 0.3
    
    L1_REGULARIZATION: float = 0.001
    L2_REGULARIZATION: float = 0.001
    
    EARLY_STOPPING_PATIENCE: int = 15
    REDUCE_LR_PATIENCE: int = 8
    MIN_LR: float = 0.00001
    
    # Cross-validation
    USE_WALK_FORWARD: bool = True
    N_SPLITS: int = 5
    
    # Ensemble
    ENABLE_ENSEMBLE: bool = True
    N_MODELS: int = 3
    
    # Data features
    TECHNICAL_INDICATORS: List[str] = None
    
    def __post_init__(self):
        if self.TECHNICAL_INDICATORS is None:
            self.TECHNICAL_INDICATORS = [
                'SMA_7', 'SMA_21', 'SMA_50', 'SMA_200',
                'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Upper', 'BB_Middle', 'BB_Lower',
                'ATR', 'ADX', 'CCI',
                'Stochastic_K', 'Stochastic_D',
                'OBV', 'Volume_MA'
            ]

config = SystemConfig()