"""
Model evaluation metrics.
Fixed version.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple


class MetricsCalculator:
    """Calculates evaluation metrics for model predictions."""
    
    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Calculate all metrics: RMSE, MAE, MAPE, R².
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Tuple of (rmse, mae, mape, r2)
        """
        try:
            mask = ~(
                np.isnan(y_true) | np.isnan(y_pred) |
                np.isinf(y_true) | np.isinf(y_pred)
            )
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return 0.0, 0.0, 0.0, 0.0
            
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            mape = MetricsCalculator._calculate_mape(y_true_clean, y_pred_clean)
            r2 = r2_score(y_true_clean, y_pred_clean)
            
            return rmse, mae, mape, r2
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value
        """
        mask = y_true != 0
        if not mask.any():
            return 0.0
        
        mape = np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
        
        return mape
    
    @staticmethod
    def calculate_directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate directional accuracy (up/down prediction accuracy).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy percentage
        """
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        accuracy = np.mean(true_direction == pred_direction) * 100
        
        return accuracy