"""
Advanced data fetcher with multiple data sources.
Inspired by Renaissance Technologies approach.
Fixed version with comprehensive error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
import streamlit as st
import warnings

warnings.filterwarnings('ignore')


class MultiSourceDataFetcher:
    """Fetch data from multiple sources with robust error handling."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_market_data(ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch traditional market data with validation."""
        try:
            if not ticker or not ticker.strip():
                st.error("Ticker cannot be empty")
                return None
            
            # Add buffer for weekends/holidays
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days * 1.5))
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data is None or data.empty:
                st.error(f"No data available for ticker: {ticker}")
                return None
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                st.error("Data missing required columns")
                return None
            
            # Clean data
            data = data.dropna()
            
            # Remove invalid rows
            data = data[
                (data['Open'] > 0) & 
                (data['High'] > 0) & 
                (data['Low'] > 0) & 
                (data['Close'] > 0) &
                (data['Volume'] >= 0)
            ]
            
            if len(data) < days * 0.5:  # At least 50% of requested days
                st.warning(f"Limited data available: {len(data)} days")
            
            return data.tail(days)
            
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_fundamental_data(ticker: str) -> Dict:
        """Fetch fundamental data with safe defaults."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Safe get with defaults
            fundamentals = {
                'market_cap': info.get('marketCap', 0) or 0,
                'pe_ratio': info.get('trailingPE', 0) or 0,
                'forward_pe': info.get('forwardPE', 0) or 0,
                'peg_ratio': info.get('pegRatio', 0) or 0,
                'price_to_book': info.get('priceToBook', 0) or 0,
                'debt_to_equity': info.get('debtToEquity', 0) or 0,
                'roe': info.get('returnOnEquity', 0) or 0,
                'roa': info.get('returnOnAssets', 0) or 0,
                'profit_margin': info.get('profitMargins', 0) or 0,
                'operating_margin': info.get('operatingMargins', 0) or 0,
                'revenue_growth': info.get('revenueGrowth', 0) or 0,
                'earnings_growth': info.get('earningsGrowth', 0) or 0,
                'current_ratio': info.get('currentRatio', 0) or 0,
                'quick_ratio': info.get('quickRatio', 0) or 0,
                'beta': info.get('beta', 1.0) or 1.0,
                'dividend_yield': info.get('dividendYield', 0) or 0,
                'payout_ratio': info.get('payoutRatio', 0) or 0,
            }
            
            # Clean infinite/NaN values
            for key, value in fundamentals.items():
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    fundamentals[key] = 0
            
            return fundamentals
            
        except Exception as e:
            # Return empty dict instead of None
            return {}
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_options_data(ticker: str) -> Optional[pd.DataFrame]:
        """Fetch options data for implied volatility analysis."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = stock.options
            if not expirations or len(expirations) == 0:
                return None
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            
            if opt_chain is None:
                return None
            
            # Combine calls and puts safely
            calls = opt_chain.calls[['strike', 'lastPrice', 'impliedVolatility', 'volume']].copy()
            puts = opt_chain.puts[['strike', 'lastPrice', 'impliedVolatility', 'volume']].copy()
            
            calls['type'] = 'call'
            puts['type'] = 'put'
            
            options = pd.concat([calls, puts], ignore_index=True)
            
            # Clean data
            options = options.dropna()
            options = options[options['impliedVolatility'] > 0]
            
            return options if not options.empty else None
            
        except Exception:
            return None
    
    @staticmethod
    def calculate_implied_volatility_metrics(options_df: pd.DataFrame) -> Dict:
        """Calculate IV metrics from options data with safe defaults."""
        if options_df is None or options_df.empty:
            return {}
        
        try:
            calls = options_df[options_df['type'] == 'call']
            puts = options_df[options_df['type'] == 'put']
            
            # Safe calculations
            avg_iv = options_df['impliedVolatility'].mean() if len(options_df) > 0 else 0
            call_iv = calls['impliedVolatility'].mean() if len(calls) > 0 else 0
            put_iv = puts['impliedVolatility'].mean() if len(puts) > 0 else 0
            
            call_vol = calls['volume'].sum() if len(calls) > 0 else 0
            put_vol = puts['volume'].sum() if len(puts) > 0 else 0
            
            metrics = {
                'avg_iv': avg_iv,
                'call_iv': call_iv,
                'put_iv': put_iv,
                'iv_skew': put_iv - call_iv,
                'put_call_ratio': put_vol / (call_vol + 1) if call_vol > 0 else 0,
            }
            
            # Clean values
            for key, value in metrics.items():
                if np.isnan(value) or np.isinf(value):
                    metrics[key] = 0
            
            return metrics
            
        except Exception:
            return {}
    
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_analyst_data(ticker: str) -> Dict:
        """Fetch analyst recommendations and price targets."""
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            
            if recommendations is None or recommendations.empty:
                return {}
            
            recent = recommendations.tail(10)
            
            analyst_data = {
                'strong_buy': len(recent[recent['To Grade'] == 'Strong Buy']),
                'buy': len(recent[recent['To Grade'] == 'Buy']),
                'hold': len(recent[recent['To Grade'] == 'Hold']),
                'sell': len(recent[recent['To Grade'] == 'Sell']),
                'strong_sell': len(recent[recent['To Grade'] == 'Strong Sell']),
            }
            
            # Calculate consensus
            total = sum(analyst_data.values())
            if total > 0:
                analyst_data['consensus_score'] = (
                    (analyst_data['strong_buy'] * 2 + analyst_data['buy']) -
                    (analyst_data['sell'] + analyst_data['strong_sell'] * 2)
                ) / total
            else:
                analyst_data['consensus_score'] = 0
            
            return analyst_data
            
        except Exception:
            return {}


class StockDataFetcher:
    """Legacy wrapper for compatibility."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch(ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch stock data (legacy method)."""
        return MultiSourceDataFetcher.fetch_market_data(ticker, days)