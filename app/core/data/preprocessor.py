"""
Data preprocessing utilities.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """Preprocesses data for LSTM training."""
    
    @staticmethod
    def create_sequences(
        data: np.ndarray, 
        lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM.
        
        Args:
            data: Scaled price data
            lookback: Number of time steps to look back
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def scale_data(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Scale data using MinMaxScaler.
        
        Args:
            data: Raw price data
            
        Returns:
            Tuple of (scaled_data, scaler)
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        return scaled_data, scaler
    
    @staticmethod
    def split_data(
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input sequences
            y: Target values
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        train_size = int(train_split * len(X))
        val_size = int(val_split * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    @staticmethod
    def prepare_for_training(
        data: pd.DataFrame,
        lookback: int,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> dict:
        """
        Complete data preparation pipeline.
        
        Args:
            data: Stock data DataFrame
            lookback: Lookback period
            train_split: Training data proportion
            val_split: Validation data proportion
            
        Returns:
            Dictionary containing all prepared data and scaler
        """
        # Extract close prices
        close_prices = data['Close'].values
        
        # Scale data
        scaled_data, scaler = DataPreprocessor.scale_data(close_prices)
        
        # Create sequences
        X, y = DataPreprocessor.create_sequences(scaled_data, lookback)
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = DataPreprocessor.split_data(
            X, y, train_split, val_split
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'scaled_data': scaled_data
        }
