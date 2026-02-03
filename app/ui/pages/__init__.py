"""UI Pages module."""

from .landing import LandingPage
from .data_stats import DataStatsPage
from .technical_analysis import TechnicalAnalysisPage
from .model_training import ModelTrainingPage
from .prediction import PredictionPage
from .export import ExportPage

__all__ = [
    'LandingPage',
    'DataStatsPage',
    'TechnicalAnalysisPage',
    'ModelTrainingPage',
    'PredictionPage',
    'ExportPage'
]
