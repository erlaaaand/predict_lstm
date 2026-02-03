"""
Data validation utilities.
"""

from typing import Tuple
import pandas as pd
import numpy as np


class DataValidator:
    """Validates stock data for training."""
    
    @staticmethod
    def validate_for_training(
        data: pd.DataFrame, 
        lookback: int
    ) -> Tuple[bool, str]:
        """
        Validate data for LSTM training.
        
        Args:
            data: Stock data DataFrame
            lookback: Lookback period for sequences
            
        Returns:
            Tuple of (is_valid, message)
        """
        if data is None or len(data) == 0:
            return False, "Data tidak tersedia"
        
        min_required = lookback + 50  # Minimal data untuk train/test split
        if len(data) < min_required:
            return False, (
                f"Data tidak cukup. Minimal {min_required} data diperlukan, "
                f"hanya tersedia {len(data)}"
            )
        
        if 'Close' not in data.columns:
            return False, "Kolom 'Close' tidak ditemukan dalam data"
        
        if data['Close'].isna().any():
            return False, "Terdapat nilai NaN dalam data Close price"
        
        if (data['Close'] <= 0).any():
            return False, "Terdapat nilai negatif atau nol dalam data Close price"
        
        return True, "Data valid"
    
    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare stock data.
        
        Args:
            data: Raw stock data
            
        Returns:
            Cleaned DataFrame
        """
        # Remove NaN values
        data = data.dropna()
        
        # Forward fill then backward fill any remaining NaN - FIXED deprecated method
        data = data.ffill().bfill()
        
        # Ensure positive prices
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols:
            if col in data.columns:
                # Replace negative or zero values with minimum positive value
                min_positive = data[col][data[col] > 0].min() if (data[col] > 0).any() else 1.0
                data[col] = data[col].apply(lambda x: min_positive if x <= 0 else x)
        
        return data
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """
        Validate ticker format.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not ticker or ticker.strip() == "":
            return False, "Ticker tidak boleh kosong"
        
        if len(ticker) < 2:
            return False, "Ticker terlalu pendek"
        
        if len(ticker) > 10:
            return False, "Ticker terlalu panjang"
        
        return True, "Ticker valid"