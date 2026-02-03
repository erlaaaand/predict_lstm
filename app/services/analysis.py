"""
Analysis service - orchestrates the complete pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional

from app.core.data.fetcher import MultiSourceDataFetcher
from app.core.data.indicators import AdvancedIndicators
from app.core.data.preprocessor import DataPreprocessor
from app.core.models.builder import RegularizedLSTMBuilder, EnsembleModel

class AnalysisService:
    """Orchestrate complete analysis pipeline."""
    
    @staticmethod
    def fetch_all_data(ticker: str, days: int) -> Optional[Dict]:
        """Fetch all available data sources."""
        with st.spinner("Fetching market data..."):
            market_data = MultiSourceDataFetcher.fetch_market_data(ticker, days)
        
        if market_data is None:
            return None
        
        # Add technical indicators
        with st.spinner("Calculating technical indicators..."):
            market_data = AdvancedIndicators.calculate_all(market_data)
        
        # Fetch fundamental data
        fundamental_data = MultiSourceDataFetcher.fetch_fundamental_data(ticker)
        
        # Fetch options data
        options_data = MultiSourceDataFetcher.fetch_options_data(ticker)
        iv_metrics = MultiSourceDataFetcher.calculate_implied_volatility_metrics(options_data)
        
        # Fetch analyst data
        analyst_data = MultiSourceDataFetcher.fetch_analyst_data(ticker)
        
        return {
            'market_data': market_data,
            'fundamental_data': fundamental_data,
            'iv_metrics': iv_metrics,
            'analyst_data': analyst_data,
            'options_data': options_data
        }
    
    @staticmethod
    def train_model(
        data: pd.DataFrame,
        config: Dict,
        fundamental_data: Dict = None
    ) -> Optional[Dict]:
        """Train model with regularization."""
        
        # Prepare data
        with st.spinner("Preprocessing data..."):
            prep_data = DataPreprocessor.prepare_for_training(
                data,
                config['lookback'],
                fundamental_data
            )
        
        if len(prep_data['X_train']) < 50:
            st.error("Insufficient training data")
            return None
        
        # Build and train
        input_shape = (prep_data['X_train'].shape[1], prep_data['X_train'].shape[2])
        
        if config.get('use_ensemble', False):
            with st.spinner("Training ensemble models..."):
                ensemble = EnsembleModel(n_models=3)
                
                ensemble.build_ensemble(
                    input_shape=input_shape,
                    units_1=config['units_1'],
                    units_2=config['units_2'],
                    dropout=config['dropout'],
                    learning_rate=config['learning_rate']
                )
                
                histories = ensemble.train_ensemble(
                    prep_data['X_train'], prep_data['y_train'],
                    prep_data['X_val'], prep_data['y_val'],
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    verbose=0
                )
                
                model = ensemble
                history = histories[0]  # Use first model's history for display
        else:
            with st.spinner("Training model..."):
                model = RegularizedLSTMBuilder.build_bidirectional_model(
                    input_shape=input_shape,
                    units_1=config['units_1'],
                    units_2=config['units_2'],
                    dropout=config['dropout'],
                    learning_rate=config['learning_rate']
                )
                
                callbacks = RegularizedLSTMBuilder.get_callbacks()
                
                history = model.fit(
                    prep_data['X_train'], prep_data['y_train'],
                    validation_data=(prep_data['X_val'], prep_data['y_val']),
                    epochs=config['epochs'],
                    batch_size=config['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )
        
        # Calculate metrics
        metrics = AnalysisService._calculate_metrics(
            model, prep_data
        )
        
        return {
            'model': model,
            'history': history,
            'prep_data': prep_data,
            'metrics': metrics,
            'is_ensemble': config.get('use_ensemble', False)
        }
    
    @staticmethod
    def _calculate_metrics(model, prep_data: Dict) -> Dict:
        """Calculate evaluation metrics."""
        metrics = {}
        
        for split in ['train', 'val', 'test']:
            X = prep_data[f'X_{split}']
            y_true = prep_data[f'y_{split}']
            
            # Predict
            if isinstance(model, EnsembleModel):
                y_pred = model.predict(X).flatten()
            else:
                y_pred = model.predict(X, verbose=0).flatten()
            
            # Calculate metrics
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            # MAPE
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
            
            # R2
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            metrics[split] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            }
        
        return metrics
    
    @staticmethod
    def make_predictions(
        model,
        data: pd.DataFrame,
        prep_data: Dict,
        n_days: int,
        is_ensemble: bool = False
    ) -> Dict:
        """Make future predictions."""
        
        # Get last sequence
        last_sequence = prep_data['X_test'][-1:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_days):
            # Predict
            if is_ensemble:
                pred = model.predict(current_sequence)
            else:
                pred = model.predict(current_sequence, verbose=0)
            
            predictions.append(pred[0, 0])
            
            # Update sequence (simplified - assumes same feature pattern)
            new_row = current_sequence[0, -1, :].copy()
            new_row[0] = pred[0, 0]  # Update close price
            
            current_sequence = np.concatenate([
                current_sequence[:, 1:, :],
                new_row.reshape(1, 1, -1)
            ], axis=1)
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform
        target_scaler = prep_data['target_scaler']
        predictions = target_scaler.inverse_transform(predictions).flatten()
        
        # Generate dates
        last_date = data.index[-1]
        future_dates = pd.bdate_range(start=last_date, periods=n_days+1)[1:]
        
        return {
            'predictions': predictions,
            'dates': future_dates,
            'last_price': data['Close'].iloc[-1]
        }