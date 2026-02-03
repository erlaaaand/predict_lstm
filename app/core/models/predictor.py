"""
Model prediction utilities.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from typing import Optional
from datetime import timedelta


class ModelPredictor:
    """Handles model predictions."""
    
    @staticmethod
    def predict_future(
        model: Sequential,
        last_sequence: np.ndarray,
        scaler: MinMaxScaler,
        n_days: int,
        lookback: int
    ) -> Optional[np.ndarray]:
        """
        Predict future stock prices.
        
        Args:
            model: Trained LSTM model
            last_sequence: Last lookback sequence from data
            scaler: Fitted MinMaxScaler
            n_days: Number of days to predict
            lookback: Lookback period
            
        Returns:
            Array of predicted prices or None if error
        """
        try:
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(n_days):
                # Reshape for prediction
                pred_input = current_sequence.reshape(1, lookback, 1)
                
                # Make prediction
                pred = model.predict(pred_input, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], pred[0, 0])
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            )
            
            return predictions.flatten()
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
    
    @staticmethod
    def generate_future_dates(
        last_date: pd.Timestamp,
        n_days: int
    ) -> pd.DatetimeIndex:
        """
        Generate future trading dates (business days only).
        
        Args:
            last_date: Last date in historical data
            n_days: Number of days to generate
            
        Returns:
            DatetimeIndex with future business days
        """
        start_date = last_date + timedelta(days=1)
        return pd.bdate_range(start=start_date, periods=n_days)
    
    @staticmethod
    def create_prediction_dataframe(
        dates: pd.DatetimeIndex,
        predictions: np.ndarray,
        last_actual_price: float
    ) -> pd.DataFrame:
        """
        Create DataFrame with predictions and metrics.
        
        Args:
            dates: Future dates
            predictions: Predicted prices
            last_actual_price: Last actual price from data
            
        Returns:
            DataFrame with predictions and changes
        """
        df = pd.DataFrame({
            'Date': dates,
            'Predicted_Price': predictions
        })
        df.set_index('Date', inplace=True)
        
        # Calculate daily changes
        df['Daily_Change'] = df['Predicted_Price'].diff()
        df['Daily_Change_%'] = df['Predicted_Price'].pct_change() * 100
        
        # Set first row values
        df.loc[df.index[0], 'Daily_Change'] = predictions[0] - last_actual_price
        df.loc[df.index[0], 'Daily_Change_%'] = (
            (predictions[0] - last_actual_price) / last_actual_price * 100
        )
        
        return df
    
    @staticmethod
    def calculate_prediction_stats(
        predictions: np.ndarray,
        last_actual_price: float
    ) -> dict:
        """
        Calculate statistics for predictions.
        
        Args:
            predictions: Array of predicted prices
            last_actual_price: Last actual price
            
        Returns:
            Dictionary with prediction statistics
        """
        return {
            'mean': predictions.mean(),
            'std': predictions.std(),
            'min': predictions.min(),
            'max': predictions.max(),
            'final': predictions[-1],
            'expected_return': ((predictions[-1] - last_actual_price) / last_actual_price) * 100,
            'range': predictions.max() - predictions.min(),
            'cv': (predictions.std() / predictions.mean()) * 100  # Coefficient of variation
        }
