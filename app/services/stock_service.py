"""
Stock analysis service - orchestrates data fetching, processing, and model training.
Fixed version with compatibility wrapper.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from app.core.data import (
    DataValidator,
)
from app.core.data.fetcher import StockDataFetcher, MultiSourceDataFetcher
from app.core.data.indicators import TechnicalIndicators, AdvancedIndicators
from app.core.data.preprocessor import DataPreprocessor
from app.core.models import LSTMModelBuilder, ModelTrainer, ModelPredictor
from app.core.utils import MetricsCalculator
from config.settings import config


class StockAnalysisService:
    """Orchestrates the entire stock analysis workflow."""
    
    @staticmethod
    def fetch_and_prepare_data(
        ticker: str,
        n_days: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data and add technical indicators.
        
        Args:
            ticker: Stock ticker symbol
            n_days: Number of trading days to fetch
            
        Returns:
            DataFrame with stock data and indicators or None
        """
        is_valid, message = DataValidator.validate_ticker(ticker)
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        
        data = StockDataFetcher.fetch(ticker, n_days)
        if data is None:
            return None
        
        data = DataValidator.clean_data(data)
        data = TechnicalIndicators.add_all_indicators(data)
        
        return data
    
    @staticmethod
    def train_model(
        data: pd.DataFrame,
        model_config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Train LSTM model on stock data.
        
        Args:
            data: Stock data DataFrame
            model_config: Model configuration parameters
            
        Returns:
            Dictionary with trained model and results or None
        """
        is_valid, message = DataValidator.validate_for_training(
            data, model_config['lookback']
        )
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        
        with st.spinner("Mempersiapkan data untuk training..."):
            try:
                prepared_data = DataPreprocessor.prepare_for_training(
                    data,
                    model_config['lookback'],
                    config.TRAIN_SPLIT,
                    config.VAL_SPLIT
                )
            except Exception as e:
                st.error(f"Preprocessing error: {str(e)}")
                return None
        
        # Get n_features from prepared data
        n_features = prepared_data.get('n_features', 1)
        
        model = LSTMModelBuilder.build(
            lookback=model_config['lookback'],
            units_1=model_config.get('lstm_units_1', 128),
            units_2=model_config.get('lstm_units_2', 64),
            dropout=model_config.get('dropout', 0.2),
            model_type=model_config.get('model_type', 'Bidirectional LSTM'),
            learning_rate=model_config.get('learning_rate', 0.001),
            n_features=n_features
        )
        
        if model is None:
            st.error("Failed to build model")
            return None
        
        with st.spinner("🎯 Melatih model LSTM..."):
            history = ModelTrainer.train(
                model=model,
                X_train=prepared_data['X_train'],
                y_train=prepared_data['y_train'],
                X_val=prepared_data['X_val'],
                y_val=prepared_data['y_val'],
                epochs=model_config.get('epochs', 100),
                batch_size=model_config.get('batch_size', 32)
            )
        
        if history is None:
            return None
        
        metrics = StockAnalysisService._evaluate_model(
            model,
            prepared_data,
            prepared_data['target_scaler']
        )
        
        return {
            'model': model,
            'history': history,
            'scaler': prepared_data['target_scaler'],
            'metrics': metrics,
            'prepared_data': prepared_data
        }
    
    @staticmethod
    def _evaluate_model(
        model,
        prepared_data: Dict[str, Any],
        scaler
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model on train, val, and test sets."""
        metrics = {}
        
        for dataset in ['train', 'val', 'test']:
            X = prepared_data[f'X_{dataset}']
            y = prepared_data[f'y_{dataset}']
            
            predictions = model.predict(X, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            y_actual = scaler.inverse_transform(y.reshape(-1, 1))
            
            rmse, mae, mape, r2 = MetricsCalculator.calculate_all(
                y_actual.flatten(),
                predictions.flatten()
            )
            
            metrics[dataset] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }
        
        return metrics
    
    @staticmethod
    def make_predictions(
        model,
        data: pd.DataFrame,
        scaler,
        lookback: int,
        n_days: int
    ) -> Optional[Dict[str, Any]]:
        """
        Make future price predictions.
        
        Args:
            model: Trained model
            data: Historical stock data
            scaler: Fitted scaler
            lookback: Lookback period
            n_days: Number of days to predict
            
        Returns:
            Dictionary with predictions and metadata
        """
        close_prices = data['Close'].values
        scaled_data = scaler.transform(close_prices.reshape(-1, 1))
        last_sequence = scaled_data[-lookback:]
        
        predictions = ModelPredictor.predict_future(
            model, last_sequence, scaler, n_days, lookback
        )
        
        if predictions is None:
            return None
        
        last_date = data.index[-1]
        future_dates = ModelPredictor.generate_future_dates(last_date, n_days)
        
        last_actual_price = data['Close'].iloc[-1]
        pred_df = ModelPredictor.create_prediction_dataframe(
            future_dates, predictions, last_actual_price
        )
        
        stats = ModelPredictor.calculate_prediction_stats(
            predictions, last_actual_price
        )
        
        return {
            'predictions': predictions,
            'dataframe': pred_df,
            'stats': stats,
            'dates': future_dates
        }