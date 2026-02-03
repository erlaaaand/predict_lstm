"""
Technical indicators calculator.
"""

import pandas as pd
import numpy as np
import streamlit as st


class TechnicalIndicators:
    """Calculates technical indicators for stock data."""
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to stock data.
        
        Args:
            data: Stock data DataFrame
            
        Returns:
            DataFrame with added indicators
        """
        try:
            data = TechnicalIndicators._add_moving_averages(data)
            data = TechnicalIndicators._add_returns_and_volatility(data)
            data = TechnicalIndicators._add_rsi(data)
            data = TechnicalIndicators._add_volume_indicators(data)
            
        except Exception as e:
            st.warning(f"⚠️ Beberapa indikator teknikal tidak dapat dihitung: {str(e)}")
        
        return data
    
    @staticmethod
    def _add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        data['MA7'] = data['Close'].rolling(window=7, min_periods=1).mean()
        data['MA21'] = data['Close'].rolling(window=21, min_periods=1).mean()
        data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        return data
    
    @staticmethod
    def _add_returns_and_volatility(data: pd.DataFrame) -> pd.DataFrame:
        """Add returns and volatility indicators."""
        data['Returns'] = data['Close'].pct_change().fillna(0)
        data['Volatility'] = data['Returns'].rolling(window=20, min_periods=1).std()
        return data
    
    @staticmethod
    def _add_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        
        Args:
            data: Stock data
            period: RSI period (default: 14)
        """
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    @staticmethod
    def _add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        if 'Volume' in data.columns:
            data['Volume_MA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA'].replace(0, 1)
        return data
    
    @staticmethod
    def add_macd(data: pd.DataFrame, 
                 fast: int = 12, 
                 slow: int = 26, 
                 signal: int = 9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Stock data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
        
        data['MACD'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        return data
    
    @staticmethod
    def add_bollinger_bands(data: pd.DataFrame, 
                           period: int = 20, 
                           std_dev: int = 2) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        Args:
            data: Stock data
            period: Moving average period
            std_dev: Number of standard deviations
        """
        ma = data['Close'].rolling(window=period, min_periods=1).mean()
        std = data['Close'].rolling(window=period, min_periods=1).std()
        
        data['BB_Middle'] = ma
        data['BB_Upper'] = ma + (std * std_dev)
        data['BB_Lower'] = ma - (std * std_dev)
        
        return data
