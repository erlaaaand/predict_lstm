"""
Advanced data preprocessing with feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict

class DataPreprocessor:
    """Preprocess data with feature engineering."""
    
    @staticmethod
    def prepare_features(
        data: pd.DataFrame,
        fundamental_data: Dict = None
    ) -> pd.DataFrame:
        """Prepare feature matrix including fundamentals."""
        df = data.copy()
        
        # Select technical features
        feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Add fundamental features if available
        if fundamental_data:
            for key, value in fundamental_data.items():
                if value and not np.isnan(value):
                    df[key] = value
        
        # Forward fill and drop NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.dropna()
        
        return df
    
    @staticmethod
    def create_sequences(
        data: np.ndarray,
        lookback: int,
        target_col_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, target_col_idx])
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def scale_data(
        data: np.ndarray,
        scaler=None
    ) -> Tuple[np.ndarray, object]:
        """Scale data using RobustScaler (resistant to outliers)."""
        if scaler is None:
            scaler = RobustScaler()
            scaled = scaler.fit_transform(data)
        else:
            scaled = scaler.transform(data)
        
        return scaled, scaler
    
    @staticmethod
    def split_data(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        val_size: float = 0.15
    ) -> Tuple:
        """Split data chronologically."""
        n = len(X)
        
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    @staticmethod
    def prepare_for_training(
        data: pd.DataFrame,
        lookback: int,
        fundamental_data: Dict = None
    ) -> Dict:
        """Complete preprocessing pipeline."""
        # Prepare features
        df = DataPreprocessor.prepare_features(data, fundamental_data)
        
        # Extract target (Close price)
        target = df['Close'].values.reshape(-1, 1)
        
        # Extract all features
        features = df.values
        
        # Scale
        scaled_data, scaler = DataPreprocessor.scale_data(features)
        
        # Target scaler (for inverse transform)
        target_scaled, target_scaler = DataPreprocessor.scale_data(target)
        
        # Create sequences
        X, y = DataPreprocessor.create_sequences(scaled_data, lookback, target_col_idx=df.columns.get_loc('Close'))
        
        # Split
        X_train, y_train, X_val, y_val, X_test, y_test = DataPreprocessor.split_data(X, y)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'target_scaler': target_scaler,
            'feature_names': df.columns.tolist()
        }