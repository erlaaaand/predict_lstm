import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Fungsi utility
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, n_trading_days):
    """
    Mengambil data saham dengan jumlah hari perdagangan yang tepat
    """
    try:
        # Estimasi periode untuk mendapatkan n_trading_days
        # Asumsi: rata-rata 252 hari trading dalam setahun (365 hari)
        factor = 1.5  # Faktor pengaman untuk memastikan data cukup
        estimated_calendar_days = int(n_trading_days * 365/252 * factor)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=estimated_calendar_days)
        
        # Download data
        with st.spinner(f"Mengunduh data {ticker}..."):
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True,  # Auto adjust untuk split dan dividen
                progress=False,
                threads=False  # Disable threading untuk stabilitas
            )
            
            data.index = pd.to_datetime(data.index)
        
        if data.empty:
            st.error(f"Tidak ada data untuk ticker {ticker}")
            return None
        
        # Handle multi-level columns jika ada
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Clean data
        data = data.dropna()
        
        # Pastikan kita punya cukup data
        if len(data) < n_trading_days:
            # Jika data kurang, ambil periode lebih panjang
            extended_days = estimated_calendar_days * 2
            start_date = end_date - timedelta(days=extended_days)
            
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True,
                progress=False,
                threads=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.dropna()
        
        # Ambil tepat n_trading_days terakhir
        data = data.tail(n_trading_days)
        
        # Validasi data
        if len(data) < n_trading_days:
            st.warning(f"⚠️ Hanya tersedia {len(data)} hari perdagangan untuk {ticker}")
        
        # Tambahkan technical indicators dengan error handling
        try:
            # Simple Moving Averages
            data['MA7'] = data['Close'].rolling(window=7, min_periods=1).mean()
            data['MA21'] = data['Close'].rolling(window=21, min_periods=1).mean()
            data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            
            # Returns dan Volatility
            data['Returns'] = data['Close'].pct_change().fillna(0)
            data['Volatility'] = data['Returns'].rolling(window=20, min_periods=1).std()
            
            # RSI (Relative Strength Index)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Volume indicators
            data['Volume_MA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA'].replace(0, 1)
            
        except Exception as e:
            st.warning(f"⚠️ Beberapa indikator teknikal tidak dapat dihitung: {str(e)}")
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error mengambil data: {str(e)}")
        st.info("💡 Tips: Periksa koneksi internet dan validitas ticker saham")
        return None

def validate_data_for_training(data, lookback):
    """
    Validasi data sebelum training
    """
    if data is None:
        return False, "Data tidak tersedia"
    
    if len(data) < lookback + 20:  # Minimal data untuk train/test split
        return False, f"Data tidak cukup. Minimal {lookback + 20} data diperlukan, hanya tersedia {len(data)}"
    
    if data['Close'].isna().any():
        return False, "Terdapat nilai NaN dalam data Close price"
    
    if (data['Close'] <= 0).any():
        return False, "Terdapat nilai negatif atau nol dalam data Close price"
    
    return True, "Data valid"

def create_sequences(data, lookback):
    """
    Create sequences for LSTM with error handling
    """
    try:
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"Error creating sequences: {str(e)}")
        return None, None
