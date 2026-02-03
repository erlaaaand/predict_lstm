"""
Configuration settings for the stock prediction application.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # App metadata
    APP_TITLE: str = "Stock Price Prediction - LSTM"
    APP_ICON: str = "📈"
    LAYOUT: str = "wide"
    
    # Page config
    SIDEBAR_STATE: str = "expanded"
    
    # Cache settings
    CACHE_TTL: int = 3600  # 1 hour
    
    # Data settings
    MIN_DATA_DAYS: int = 90
    MAX_DATA_DAYS: int = 730
    DEFAULT_DATA_DAYS: int = 365
    TRADING_DAYS_PER_YEAR: int = 252
    
    # Prediction settings
    MIN_PREDICT_DAYS: int = 1
    MAX_PREDICT_DAYS: int = 60
    DEFAULT_PREDICT_DAYS: int = 14
    
    # Model settings
    MIN_LOOKBACK: int = 10
    MAX_LOOKBACK: int = 90
    DEFAULT_LOOKBACK: int = 60
    
    MIN_LSTM_UNITS: int = 64
    MAX_LSTM_UNITS: int = 256
    DEFAULT_LSTM_UNITS_1: int = 128
    DEFAULT_LSTM_UNITS_2: int = 64
    
    MIN_DROPOUT: float = 0.1
    MAX_DROPOUT: float = 0.5
    DEFAULT_DROPOUT: float = 0.2
    
    MIN_EPOCHS: int = 50
    MAX_EPOCHS: int = 200
    DEFAULT_EPOCHS: int = 100
    
    BATCH_SIZES: List[int] = None
    DEFAULT_BATCH_SIZE: int = 32
    
    LEARNING_RATES: List[float] = None
    DEFAULT_LEARNING_RATE: float = 0.001
    
    # Model types
    MODEL_TYPES: List[str] = None
    
    # Training settings
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 20
    REDUCE_LR_PATIENCE: int = 10
    MIN_LEARNING_RATE: float = 0.00001
    
    # Visualization settings
    HISTORICAL_DISPLAY_DAYS: int = 60
    CHART_HEIGHT: int = 600
    
    # Sample tickers
    SAMPLE_TICKERS: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        """Initialize list and dict fields."""
        if self.BATCH_SIZES is None:
            self.BATCH_SIZES = [16, 32, 64, 128]
        
        if self.LEARNING_RATES is None:
            self.LEARNING_RATES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        
        if self.MODEL_TYPES is None:
            self.MODEL_TYPES = ["Standard LSTM", "Bidirectional LSTM", "Deep LSTM"]
        
        if self.SAMPLE_TICKERS is None:
            self.SAMPLE_TICKERS = {
                'Banking': {'ticker': 'BBCA.JK', 'name': 'Bank Central Asia'},
                'Telekomunikasi': {'ticker': 'TLKM.JK', 'name': 'Telkom Indonesia'},
                'Consumer': {'ticker': 'ICBP.JK', 'name': 'Indofood CBP'},
                'Mining': {'ticker': 'ADRO.JK', 'name': 'Adaro Energy'},
                'Property': {'ticker': 'BSDE.JK', 'name': 'Bumi Serpong Damai'}
            }


# Global config instance
config = AppConfig()
