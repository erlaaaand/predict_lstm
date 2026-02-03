"""
Advanced technical indicators calculator.
Comprehensive set of indicators for quantitative analysis.
Fixed version with safe operations and error handling.
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class AdvancedIndicators:
    """Calculate comprehensive technical indicators with safe operations."""
    
    @staticmethod
    def calculate_all(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with error handling."""
        try:
            df = data.copy()
            
            # Ensure minimum data
            if len(df) < 5:
                return df
            
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
            
            # Clean NaN and Inf
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return data
    
    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        try:
            for period in [7, 21, 50, 200]:
                if len(df) >= period:
                    df[f'SMA_{period}'] = df['Close'].rolling(
                        window=period, min_periods=1
                    ).mean()
                else:
                    df[f'SMA_{period}'] = df['Close']
            
            for period in [12, 26]:
                if len(df) >= period:
                    df[f'EMA_{period}'] = df['Close'].ewm(
                        span=period, adjust=False, min_periods=1
                    ).mean()
                else:
                    df[f'EMA_{period}'] = df['Close']
            
        except Exception:
            pass
        
        return df
    
    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        try:
            if len(df) < period:
                df['RSI'] = 50  # Neutral value
                return df
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(
                window=period, min_periods=1
            ).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(
                window=period, min_periods=1
            ).mean()
            
            # Avoid division by zero
            loss = loss.replace(0, 1e-10)
            
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Clip to valid range
            df['RSI'] = df['RSI'].clip(0, 100)
            
        except Exception:
            df['RSI'] = 50
        
        return df
    
    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators."""
        try:
            if len(df) < 26:
                df['MACD'] = 0
                df['MACD_Signal'] = 0
                df['MACD_Hist'] = 0
                return df
            
            ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
        except Exception:
            df['MACD'] = 0
            df['MACD_Signal'] = 0
            df['MACD_Hist'] = 0
        
        return df
    
    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands."""
        try:
            if len(df) < period:
                df['BB_Middle'] = df['Close']
                df['BB_Upper'] = df['Close'] * 1.02
                df['BB_Lower'] = df['Close'] * 0.98
                df['BB_Width'] = 0.04
                return df
            
            ma = df['Close'].rolling(window=period, min_periods=1).mean()
            std = df['Close'].rolling(window=period, min_periods=1).std()
            
            df['BB_Middle'] = ma
            df['BB_Upper'] = ma + (std * std_dev)
            df['BB_Lower'] = ma - (std * std_dev)
            
            # Avoid division by zero
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-10)
            
        except Exception:
            df['BB_Middle'] = df['Close']
            df['BB_Upper'] = df['Close'] * 1.02
            df['BB_Lower'] = df['Close'] * 0.98
            df['BB_Width'] = 0.04
        
        return df
    
    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        try:
            if len(df) < 2:
                df['ATR'] = 0
                return df
            
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=period, min_periods=1).mean()
            
        except Exception:
            df['ATR'] = 0
        
        return df
    
    @staticmethod
    def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index."""
        try:
            if len(df) < period:
                df['ADX'] = 20  # Neutral value
                return df
            
            plus_dm = df['High'].diff()
            minus_dm = -df['Low'].diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = AdvancedIndicators._calculate_true_range(df)
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            # Avoid division by zero
            atr = atr.replace(0, 1e-10)
            
            plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df['ADX'] = dx.rolling(window=period, min_periods=1).mean()
            
            # Clip to valid range
            df['ADX'] = df['ADX'].clip(0, 100)
            
        except Exception:
            df['ADX'] = 20
        
        return df
    
    @staticmethod
    def _calculate_true_range(df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        except Exception:
            return pd.Series([0] * len(df), index=df.index)
    
    @staticmethod
    def _add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        try:
            if len(df) < k_period:
                df['Stochastic_K'] = 50
                df['Stochastic_D'] = 50
                return df
            
            low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
            high_max = df['High'].rolling(window=k_period, min_periods=1).max()
            
            # Avoid division by zero
            denominator = high_max - low_min
            denominator = denominator.replace(0, 1e-10)
            
            df['Stochastic_K'] = 100 * (df['Close'] - low_min) / denominator
            df['Stochastic_D'] = df['Stochastic_K'].rolling(
                window=d_period, min_periods=1
            ).mean()
            
            # Clip to valid range
            df['Stochastic_K'] = df['Stochastic_K'].clip(0, 100)
            df['Stochastic_D'] = df['Stochastic_D'].clip(0, 100)
            
        except Exception:
            df['Stochastic_K'] = 50
            df['Stochastic_D'] = 50
        
        return df
    
    @staticmethod
    def _add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        try:
            if len(df) < period:
                df['CCI'] = 0
                return df
            
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            ma = tp.rolling(window=period, min_periods=1).mean()
            md = tp.rolling(window=period, min_periods=1).apply(
                lambda x: np.abs(x - x.mean()).mean(), raw=True
            )
            
            # Avoid division by zero
            denominator = 0.015 * md
            denominator = denominator.replace(0, 1e-10)
            
            df['CCI'] = (tp - ma) / denominator
            
            # Clip to reasonable range
            df['CCI'] = df['CCI'].clip(-300, 300)
            
        except Exception:
            df['CCI'] = 0
        
        return df
    
    @staticmethod
    def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        try:
            if len(df) < 2:
                df['OBV'] = 0
                return df
            
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            
            df['OBV'] = obv
            
        except Exception:
            df['OBV'] = 0
        
        return df
    
    @staticmethod
    def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        try:
            df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
            
            # Avoid division by zero
            volume_ma_safe = df['Volume_MA'].replace(0, 1e-10)
            df['Volume_Ratio'] = df['Volume'] / volume_ma_safe
            
        except Exception:
            df['Volume_MA'] = df['Volume']
            df['Volume_Ratio'] = 1.0
        
        return df
    
    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add returns and volatility measures."""
        try:
            df['Returns'] = df['Close'].pct_change().fillna(0)
            
            # Log returns with safe calculation
            close_shifted = df['Close'].shift(1)
            close_shifted = close_shifted.replace(0, 1e-10)
            df['Log_Returns'] = np.log(df['Close'] / close_shifted).fillna(0)
            
            df['Volatility_20'] = df['Returns'].rolling(window=20, min_periods=1).std()
            df['Volatility_60'] = df['Returns'].rolling(window=60, min_periods=1).std()
            
            # Replace any remaining inf/nan
            df['Returns'] = df['Returns'].replace([np.inf, -np.inf], 0).fillna(0)
            df['Log_Returns'] = df['Log_Returns'].replace([np.inf, -np.inf], 0).fillna(0)
            df['Volatility_20'] = df['Volatility_20'].replace([np.inf, -np.inf], 0).fillna(0)
            df['Volatility_60'] = df['Volatility_60'].replace([np.inf, -np.inf], 0).fillna(0)
            
        except Exception:
            df['Returns'] = 0
            df['Log_Returns'] = 0
            df['Volatility_20'] = 0
            df['Volatility_60'] = 0
        
        return df


# Legacy wrapper
class TechnicalIndicators:
    """Legacy wrapper for compatibility."""
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators (legacy method)."""
        return AdvancedIndicators.calculate_all(data)