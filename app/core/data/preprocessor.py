"""
Advanced data preprocessing with feature engineering.
Fixed version with robust error handling and validation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocess data with feature engineering and validation."""
    
    @staticmethod
    def prepare_features(
        data: pd.DataFrame,
        fundamental_data: Dict = None
    ) -> pd.DataFrame:
        """Prepare feature matrix including fundamentals."""
        try:
            df = data.copy()
            
            # Ensure minimum required columns
            if 'Close' not in df.columns:
                raise ValueError("'Close' column is required")
            
            # Select features (exclude OHLCV for now, keep only indicators)
            exclude_cols = ['Open', 'High', 'Low', 'Volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            df = df[feature_cols]
            
            # Add fundamental features if available
            if fundamental_data:
                for key, value in fundamental_data.items():
                    # Only add valid numeric values
                    if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        if value != 0:  # Skip zero values that don't add information
                            df[key] = value
            
            # Forward fill and backward fill
            df = df.ffill().bfill()
            
            # Drop any remaining NaN
            df = df.dropna()
            
            # Replace inf values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            
            # Final check
            if df.empty:
                raise ValueError("No valid data after preprocessing")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error preparing features: {str(e)}")
    
    @staticmethod
    def create_sequences(
        data: np.ndarray,
        lookback: int,
        target_col_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM with validation."""
        try:
            if len(data) < lookback + 1:
                raise ValueError(f"Insufficient data: need at least {lookback + 1} rows")
            
            X, y = [], []
            
            for i in range(lookback, len(data)):
                sequence = data[i-lookback:i]
                
                # Validate sequence
                if np.isnan(sequence).any() or np.isinf(sequence).any():
                    continue
                
                X.append(sequence)
                y.append(data[i, target_col_idx])
            
            if len(X) == 0:
                raise ValueError("No valid sequences created")
            
            X = np.array(X)
            y = np.array(y)
            
            # Final validation
            if np.isnan(X).any() or np.isinf(X).any():
                raise ValueError("Invalid values in sequences")
            
            if np.isnan(y).any() or np.isinf(y).any():
                raise ValueError("Invalid values in targets")
            
            return X, y
            
        except Exception as e:
            raise ValueError(f"Error creating sequences: {str(e)}")
    
    @staticmethod
    def scale_data(
        data: np.ndarray,
        scaler=None
    ) -> Tuple[np.ndarray, object]:
        """Scale data using RobustScaler (resistant to outliers)."""
        try:
            # Validate input
            if data is None or len(data) == 0:
                raise ValueError("Empty data for scaling")
            
            # Replace inf values
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            if scaler is None:
                scaler = RobustScaler()
                scaled = scaler.fit_transform(data)
            else:
                scaled = scaler.transform(data)
            
            # Validate output
            if np.isnan(scaled).any() or np.isinf(scaled).any():
                raise ValueError("Scaling produced invalid values")
            
            return scaled, scaler
            
        except Exception as e:
            raise ValueError(f"Error scaling data: {str(e)}")
    
    @staticmethod
    def split_data(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        val_size: float = 0.15
    ) -> Tuple:
        """Split data chronologically with validation."""
        try:
            n = len(X)
            
            if n < 50:
                raise ValueError(f"Insufficient data for splitting: {n} samples")
            
            train_end = int(n * train_size)
            val_end = int(n * (train_size + val_size))
            
            # Ensure minimum samples in each split
            if train_end < 20:
                raise ValueError("Insufficient training data")
            if val_end - train_end < 10:
                raise ValueError("Insufficient validation data")
            if n - val_end < 10:
                raise ValueError("Insufficient test data")
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            
            X_val = X[train_end:val_end]
            y_val = y[train_end:val_end]
            
            X_test = X[val_end:]
            y_test = y[val_end:]
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            raise ValueError(f"Error splitting data: {str(e)}")
    
    @staticmethod
    def prepare_for_training(
        data: pd.DataFrame,
        lookback: int,
        train_split: float = 0.7,
        val_split: float = 0.15,
        fundamental_data: Dict = None
    ) -> Dict:
        """Complete preprocessing pipeline with comprehensive validation."""
        try:
            # Validate inputs
            if data is None or data.empty:
                raise ValueError("Empty data provided")
            
            if lookback < 1:
                raise ValueError(f"Invalid lookback period: {lookback}")
            
            if len(data) < lookback + 50:
                raise ValueError(
                    f"Insufficient data: need at least {lookback + 50} rows, got {len(data)}"
                )
            
            # Prepare features
            df = DataPreprocessor.prepare_features(data, fundamental_data)
            
            # Ensure we have Close column
            if 'Close' not in df.columns:
                raise ValueError("'Close' column not found after feature preparation")
            
            # Extract features
            features = df.values
            
            # Validate features
            if np.isnan(features).any() or np.isinf(features).any():
                raise ValueError("Invalid values in features")
            
            # Scale features
            scaled_data, scaler = DataPreprocessor.scale_data(features)
            
            # Get Close column index
            close_idx = df.columns.get_loc('Close')
            
            # Create sequences
            X, y = DataPreprocessor.create_sequences(
                scaled_data, 
                lookback, 
                target_col_idx=close_idx
            )
            
            # Validate sequences
            if len(X) < 50:
                raise ValueError(f"Not enough sequences created: {len(X)}. Need at least 50.")
            
            # Split data
            X_train, y_train, X_val, y_val, X_test, y_test = DataPreprocessor.split_data(
                X, y, train_split, val_split
            )
            
            # Create separate scaler for target (Close price only)
            target_data = df[['Close']].values
            target_scaled, target_scaler = DataPreprocessor.scale_data(target_data)
            
            result = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'scaler': scaler,
                'target_scaler': target_scaler,
                'feature_names': df.columns.tolist(),
                'n_features': X.shape[2],
                'lookback': lookback
            }
            
            # Validate result
            for key in ['X_train', 'X_val', 'X_test']:
                if result[key] is None or len(result[key]) == 0:
                    raise ValueError(f"{key} is empty")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error in preprocessing pipeline: {str(e)}")