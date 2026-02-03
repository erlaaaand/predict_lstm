"""
Advanced technical indicators calculator.
Comprehensive set of indicators for quantitative analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional

class AdvancedIndicators:
    """Calculate comprehensive technical indicators."""
    
    @staticmethod
    def calculate_all(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        df = data.copy()
        
        # Moving Averages
        df = AdvancedIndicators._add_moving_averages(df)
        
        # Momentum Indicators
        df = AdvancedIndicators._add_rsi(df)
        df = AdvancedIndicators._add_macd(df)
        df = AdvancedIndicators._add_stochastic(df)
        df = AdvancedIndicators._add_cci(df)
        
        # Volatility Indicators
        df = AdvancedIndicators._add_bollinger_bands(df)
        df = AdvancedIndicators._add_atr(df)
        
        # Trend Indicators
        df = AdvancedIndicators._add_adx(df)
        
        # Volume Indicators
        df = AdvancedIndicators._add_obv(df)
        df = AdvancedIndicators._add_volume_indicators(df)
        
        # Returns and Volatility
        df = AdvancedIndicators._add_returns(df)
        
        return df
    
    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        for period in [7, 21, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
        
        for period in [12, 26]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators."""
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands."""
        ma = df['Close'].rolling(window=period, min_periods=1).mean()
        std = df['Close'].rolling(window=period, min_periods=1).std()
        
        df['BB_Middle'] = ma
        df['BB_Upper'] = ma + (std * std_dev)
        df['BB_Lower'] = ma - (std * std_dev)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        return df
    
    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index."""
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = AdvancedIndicators._calculate_true_range(df)
        
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['ADX'] = dx.rolling(window=period, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def _calculate_true_range(df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    @staticmethod
    def _add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['High'].rolling(window=k_period, min_periods=1).max()
        
        df['Stochastic_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-10)
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=d_period, min_periods=1).mean()
        
        return df
    
    @staticmethod
    def _add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(window=period, min_periods=1).mean()
        md = tp.rolling(window=period, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
        
        df['CCI'] = (tp - ma) / (0.015 * md + 1e-10)
        
        return df
    
    @staticmethod
    def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        df['OBV'] = obv
        
        return df
    
    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)
        
        return df
    
    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add returns and volatility measures."""
        df['Returns'] = df['Close'].pct_change().fillna(0)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        df['Volatility_20'] = df['Returns'].rolling(window=20, min_periods=1).std()
        df['Volatility_60'] = df['Returns'].rolling(window=60, min_periods=1).std()
        
        return df