import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics with error handling
    """
    try:
        # Remove any NaN or inf values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return 0, 0, 0, 0
        
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # MAPE with zero handling
        mape_mask = y_true_clean != 0
        if mape_mask.any():
            mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / 
                                 y_true_clean[mape_mask])) * 100
        else:
            mape = 0
        
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return rmse, mae, mape, r2
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return 0, 0, 0, 0
