"""
Stock analysis service - orchestrates data fetching, processing, and model training.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from app.core.data import (
    StockDataFetcher,
    DataValidator,
    TechnicalIndicators,
    DataPreprocessor
)
from app.core.models import LSTMModelBuilder, ModelTrainer, ModelPredictor
from app.core.utils import MetricsCalculator
from app.config import config


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
        # Validate ticker
        is_valid, message = DataValidator.validate_ticker(ticker)
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        
        # Fetch data
        data = StockDataFetcher.fetch(ticker, n_days)
        if data is None:
            return None
        
        # Clean data
        data = DataValidator.clean_data(data)
        
        # Add technical indicators
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
        # Validate data
        is_valid, message = DataValidator.validate_for_training(
            data, model_config['lookback']
        )
        if not is_valid:
            st.error(f"❌ {message}")
            return None
        
        # Prepare data for training
        with st.spinner("Mempersiapkan data untuk training..."):
            prepared_data = DataPreprocessor.prepare_for_training(
                data,
                model_config['lookback'],
                config.TRAIN_SPLIT,
                config.VAL_SPLIT
            )
        
        # Build model
        model = LSTMModelBuilder.build(
            lookback=model_config['lookback'],
            units_1=model_config['lstm_units_1'],
            units_2=model_config['lstm_units_2'],
            dropout=model_config['dropout'],
            model_type=model_config['model_type'],
            learning_rate=model_config['learning_rate']
        )
        
        if model is None:
            st.error("Failed to build model")
            return None
        
        # Train model
        with st.spinner("🎯 Melatih model LSTM..."):
            history = ModelTrainer.train(
                model=model,
                X_train=prepared_data['X_train'],
                y_train=prepared_data['y_train'],
                X_val=prepared_data['X_val'],
                y_val=prepared_data['y_val'],
                epochs=model_config['epochs'],
                batch_size=model_config['batch_size']
            )
        
        if history is None:
            return None
        
        # Calculate metrics on all sets
        metrics = StockAnalysisService._evaluate_model(
            model,
            prepared_data,
            prepared_data['scaler']
        )
        
        return {
            'model': model,
            'history': history,
            'scaler': prepared_data['scaler'],
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
            
            # Make predictions
            predictions = model.predict(X, verbose=0)
            
            # Inverse transform
            predictions = scaler.inverse_transform(predictions)
            y_actual = scaler.inverse_transform(y.reshape(-1, 1))
            
            # Calculate metrics
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
        # Get last sequence
        close_prices = data['Close'].values
        scaled_data = scaler.transform(close_prices.reshape(-1, 1))
        last_sequence = scaled_data[-lookback:]
        
        # Make predictions
        predictions = ModelPredictor.predict_future(
            model, last_sequence, scaler, n_days, lookback
        )
        
        if predictions is None:
            return None
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = ModelPredictor.generate_future_dates(last_date, n_days)
        
        # Create prediction DataFrame
        last_actual_price = data['Close'].iloc[-1]
        pred_df = ModelPredictor.create_prediction_dataframe(
            future_dates, predictions, last_actual_price
        )
        
        # Calculate statistics
        stats = ModelPredictor.calculate_prediction_stats(
            predictions, last_actual_price
        )
        
        return {
            'predictions': predictions,
            'dataframe': pred_df,
            'stats': stats,
            'dates': future_dates
        }
