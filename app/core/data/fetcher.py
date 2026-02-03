"""
Stock data fetching utilities.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from app.config import config


class StockDataFetcher:
    """Fetches stock data from Yahoo Finance."""
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL, show_spinner=False)
    def fetch(ticker: str, n_trading_days: int) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with specified number of trading days.
        
        Args:
            ticker: Stock ticker symbol
            n_trading_days: Number of trading days to fetch
            
        Returns:
            DataFrame with stock data or None if error
        """
        try:
            # Estimate calendar days needed
            factor = 1.5  # Safety factor
            estimated_days = int(
                n_trading_days * 365 / config.TRADING_DAYS_PER_YEAR * factor
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=estimated_days)
            
            # Download data
            with st.spinner(f"Mengunduh data {ticker}..."):
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )
            
            if data.empty:
                st.error(f"Tidak ada data untuk ticker {ticker}")
                return None
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Clean and validate
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            # Ensure we have enough data
            if len(data) < n_trading_days:
                # Try with extended period
                extended_days = estimated_days * 2
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
                data.index = pd.to_datetime(data.index)
            
            # Take last n_trading_days
            data = data.tail(n_trading_days)
            
            if len(data) < n_trading_days:
                st.warning(
                    f"⚠️ Hanya tersedia {len(data)} hari perdagangan untuk {ticker}"
                )
            
            return data
            
        except Exception as e:
            st.error(f"❌ Error mengambil data: {str(e)}")
            st.info("💡 Tips: Periksa koneksi internet dan validitas ticker saham")
            return None
