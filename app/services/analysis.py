"""
Analysis service - orchestrates the complete pipeline.
Fixed version with comprehensive error handling and validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings

from app.core.data.fetcher import MultiSourceDataFetcher
from app.core.data.indicators import AdvancedIndicators
from app.core.data.preprocessor import DataPreprocessor
from app.core.models.builder import RegularizedLSTMBuilder, EnsembleModel

warnings.filterwarnings('ignore')


class AnalysisService:
    """Orchestrate complete analysis pipeline with robust error handling."""
    
    @staticmethod
    def fetch_all_data(ticker: str, days: int) -> Optional[Dict]:
        """Fetch all available data sources with validation."""
        try:
            # Validate inputs
            if not ticker or not ticker.strip():
                st.error("Ticker cannot be empty")
                return None
            
            if days < 90:
                st.warning("Minimum 90 days recommended")
                days = 90
            
            # Fetch market data
            with st.spinner("Fetching market data..."):
                market_data = MultiSourceDataFetcher.fetch_market_data(ticker, days)
            
            if market_data is None or market_data.empty:
                st.error(f"Failed to fetch data for {ticker}")
                return None
            
            st.success(f"✓ Fetched {len(market_data)} days of market data")
            
            # Add technical indicators
            with st.spinner("Calculating technical indicators..."):
                market_data = AdvancedIndicators.calculate_all(market_data)
            
            st.success(f"✓ Calculated technical indicators")
            
            # Fetch fundamental data (non-blocking)
            fundamental_data = {}
            try:
                with st.spinner("Fetching fundamental data..."):
                    fundamental_data = MultiSourceDataFetcher.fetch_fundamental_data(ticker)
                if fundamental_data:
                    st.success(f"✓ Fetched {len(fundamental_data)} fundamental metrics")
            except Exception as e:
                st.warning(f"Could not fetch fundamental data: {str(e)}")
            
            # Fetch options data (non-blocking)
            options_data = None
            iv_metrics = {}
            try:
                with st.spinner("Fetching options data..."):
                    options_data = MultiSourceDataFetcher.fetch_options_data(ticker)
                    if options_data is not None:
                        iv_metrics = MultiSourceDataFetcher.calculate_implied_volatility_metrics(options_data)
                        if iv_metrics:
                            st.success(f"✓ Calculated IV metrics")
            except Exception as e:
                st.info("Options data not available")
            
            # Fetch analyst data (non-blocking)
            analyst_data = {}
            try:
                with st.spinner("Fetching analyst recommendations..."):
                    analyst_data = MultiSourceDataFetcher.fetch_analyst_data(ticker)
                    if analyst_data:
                        st.success(f"✓ Fetched analyst recommendations")
            except Exception as e:
                st.info("Analyst data not available")
            
            return {
                'market_data': market_data,
                'fundamental_data': fundamental_data,
                'iv_metrics': iv_metrics,
                'analyst_data': analyst_data,
                'options_data': options_data
            }
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    @staticmethod
    def train_model(
        data: pd.DataFrame,
        config_params: Dict,
        fundamental_data: Dict = None
    ) -> Optional[Dict]:
        """Train model with regularization and validation."""
        try:
            # Validate data
            if data is None or data.empty:
                st.error("No data provided for training")
                return None
            
            if len(data) < config_params['lookback'] + 50:
                st.error(f"Insufficient data: need at least {config_params['lookback'] + 50} rows")
                return None
            
            # Prepare data
            with st.spinner("Preprocessing data..."):
                try:
                    prep_data = DataPreprocessor.prepare_for_training(
                        data,
                        config_params['lookback'],
                        fundamental_data=fundamental_data
                    )
                except Exception as e:
                    st.error(f"Preprocessing failed: {str(e)}")
                    return None
            
            # Validate prepared data
            if len(prep_data['X_train']) < 20:
                st.error("Insufficient training data after preprocessing")
                return None
            
            st.success(f"✓ Prepared {len(prep_data['X_train'])} training samples")
            
            # Get input shape
            input_shape = (prep_data['X_train'].shape[1], prep_data['X_train'].shape[2])
            st.info(f"Input shape: {input_shape[0]} timesteps × {input_shape[1]} features")
            
            # Build and train model
            if config_params.get('use_ensemble', False):
                return AnalysisService._train_ensemble(prep_data, config_params, input_shape)
            else:
                return AnalysisService._train_single(prep_data, config_params, input_shape)
                
        except Exception as e:
            st.error(f"Training error: {str(e)}")
            return None
    
    @staticmethod
    def _train_single(prep_data: Dict, config_params: Dict, input_shape: Tuple) -> Dict:
        """Train single model."""
        try:
            with st.spinner("Building model..."):
                model = RegularizedLSTMBuilder.build_bidirectional_model(
                    input_shape=input_shape,
                    units_1=config_params['units_1'],
                    units_2=config_params['units_2'],
                    dropout=config_params['dropout'],
                    learning_rate=config_params['learning_rate']
                )
            
            st.success("✓ Model built successfully")
            
            # Train
            with st.spinner("Training model (this may take 1-3 minutes)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                callbacks = RegularizedLSTMBuilder.get_callbacks()
                
                class ProgressCallback:
                    def __init__(self, epochs, progress_bar, status_text):
                        self.epochs = epochs
                        self.progress_bar = progress_bar
                        self.status_text = status_text
                        self.epoch = 0
                    
                    def on_epoch_end(self, epoch, logs):
                        self.epoch = epoch + 1
                        progress = min(int((self.epoch / self.epochs) * 100), 100)
                        self.progress_bar.progress(progress)
                        
                        loss = logs.get('loss', 0)
                        val_loss = logs.get('val_loss', 0)
                        self.status_text.text(
                            f"Epoch {self.epoch}/{self.epochs} - "
                            f"Loss: {loss:.6f} - Val Loss: {val_loss:.6f}"
                        )
                
                from tensorflow.keras.callbacks import LambdaCallback
                
                progress_cb = ProgressCallback(config_params['epochs'], progress_bar, status_text)
                lambda_cb = LambdaCallback(on_epoch_end=progress_cb.on_epoch_end)
                callbacks.append(lambda_cb)
                
                history = model.fit(
                    prep_data['X_train'], prep_data['y_train'],
                    validation_data=(prep_data['X_val'], prep_data['y_val']),
                    epochs=config_params['epochs'],
                    batch_size=config_params['batch_size'],
                    callbacks=callbacks,
                    verbose=0
                )
                
                progress_bar.progress(100)
                status_text.success("✓ Training complete!")
            
            # Calculate metrics
            metrics = AnalysisService._calculate_metrics(model, prep_data)
            
            return {
                'model': model,
                'history': history,
                'prep_data': prep_data,
                'metrics': metrics,
                'is_ensemble': False
            }
            
        except Exception as e:
            raise ValueError(f"Error training single model: {str(e)}")
    
    @staticmethod
    def _train_ensemble(prep_data: Dict, config_params: Dict, input_shape: Tuple) -> Dict:
        """Train ensemble of models."""
        try:
            with st.spinner("Building ensemble (3 models)..."):
                ensemble = EnsembleModel(n_models=3)
                
                ensemble.build_ensemble(
                    input_shape=input_shape,
                    units_1=config_params['units_1'],
                    units_2=config_params['units_2'],
                    dropout=config_params['dropout'],
                    learning_rate=config_params['learning_rate']
                )
            
            st.success("✓ Ensemble built successfully")
            
            # Train ensemble
            with st.spinner("Training ensemble (this may take 3-5 minutes)..."):
                histories = ensemble.train_ensemble(
                    prep_data['X_train'], prep_data['y_train'],
                    prep_data['X_val'], prep_data['y_val'],
                    epochs=config_params['epochs'],
                    batch_size=config_params['batch_size'],
                    verbose=0
                )
            
            st.success("✓ Ensemble training complete!")
            
            # Calculate metrics
            metrics = AnalysisService._calculate_metrics(ensemble, prep_data)
            
            return {
                'model': ensemble,
                'history': histories[0],  # Use first model's history for display
                'prep_data': prep_data,
                'metrics': metrics,
                'is_ensemble': True
            }
            
        except Exception as e:
            raise ValueError(f"Error training ensemble: {str(e)}")
    
    @staticmethod
    def _calculate_metrics(model, prep_data: Dict) -> Dict:
        """Calculate evaluation metrics with error handling."""
        metrics = {}
        
        try:
            for split in ['train', 'val', 'test']:
                X = prep_data[f'X_{split}']
                y_true = prep_data[f'y_{split}']
                
                # Predict
                if isinstance(model, EnsembleModel):
                    y_pred = model.predict(X, verbose=0).flatten()
                else:
                    y_pred = model.predict(X, verbose=0).flatten()
                
                # Validate predictions
                if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                    st.warning(f"Invalid predictions in {split} set")
                    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Calculate metrics
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                # MAPE with zero handling
                mask = y_true != 0
                if mask.any():
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                else:
                    mape = 0
                
                # R² with safe calculation
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0
                
                metrics[split] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape),
                    'r2': float(r2)
                }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {
                'train': {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0},
                'val': {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0},
                'test': {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0}
            }
    
    @staticmethod
    def make_predictions(
        model,
        data: pd.DataFrame,
        prep_data: Dict,
        n_days: int,
        is_ensemble: bool = False
    ) -> Optional[Dict]:
        """Make future predictions with validation."""
        try:
            if n_days < 1 or n_days > 90:
                st.error("Prediction days must be between 1 and 90")
                return None
            
            with st.spinner(f"Generating {n_days}-day predictions..."):
                # Get last sequence
                last_sequence = prep_data['X_test'][-1:].copy()
                
                predictions = []
                current_sequence = last_sequence.copy()
                
                for i in range(n_days):
                    # Predict next step
                    if is_ensemble:
                        pred = model.predict(current_sequence, verbose=0)
                    else:
                        pred = model.predict(current_sequence, verbose=0)
                    
                    pred_value = float(pred[0, 0])
                    predictions.append(pred_value)
                    
                    # Update sequence
                    # Shift left and append new prediction
                    new_row = current_sequence[0, -1, :].copy()
                    new_row[0] = pred_value  # Update close price
                    
                    current_sequence = np.concatenate([
                        current_sequence[:, 1:, :],
                        new_row.reshape(1, 1, -1)
                    ], axis=1)
                
                # Validate predictions
                predictions = np.array(predictions).reshape(-1, 1)
                
                if np.isnan(predictions).any() or np.isinf(predictions).any():
                    st.error("Invalid prediction values generated")
                    return None
                
                # Inverse transform
                target_scaler = prep_data['target_scaler']
                predictions = target_scaler.inverse_transform(predictions).flatten()
                
                # Validate transformed predictions
                if np.isnan(predictions).any() or np.isinf(predictions).any():
                    st.error("Invalid values after inverse transform")
                    return None
                
                # Generate dates
                last_date = data.index[-1]
                future_dates = pd.bdate_range(start=last_date, periods=n_days+1)[1:]
                
                last_price = float(data['Close'].iloc[-1])
                
                return {
                    'predictions': predictions,
                    'dates': future_dates,
                    'last_price': last_price
                }
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None