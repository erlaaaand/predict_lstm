"""
Advanced data fetcher with multiple data sources.
Inspired by Renaissance Technologies approach.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
import streamlit as st

class MultiSourceDataFetcher:
    """Fetch data from multiple sources."""
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_market_data(ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch traditional market data."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(days * 1.5))
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if data.empty:
                return None
            
            data = data.dropna()
            return data.tail(days)
            
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data(ttl=86400, show_spinner=False)
    def fetch_fundamental_data(ticker: str) -> Optional[Dict]:
        """Fetch fundamental data (balance sheet, income statement, etc)."""
        try:
            stock = yf.Ticker(ticker)
            
            info = stock.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'roa': info.get('returnOnAssets', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0),
                'payout_ratio': info.get('payoutRatio', 0),
            }
            
            return fundamentals
            
        except Exception as e:
            st.warning(f"Could not fetch fundamental data: {str(e)}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_options_data(ticker: str) -> Optional[pd.DataFrame]:
        """Fetch options data for implied volatility analysis."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = stock.options
            if not expirations:
                return None
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            
            # Combine calls and puts
            calls = opt_chain.calls[['strike', 'lastPrice', 'impliedVolatility', 'volume']]
            puts = opt_chain.puts[['strike', 'lastPrice', 'impliedVolatility', 'volume']]
            
            calls['type'] = 'call'
            puts['type'] = 'put'
            
            options = pd.concat([calls, puts])
            
            return options
            
        except Exception:
            return None
    
    @staticmethod
    def calculate_implied_volatility_metrics(options_df: pd.DataFrame) -> Dict:
        """Calculate IV metrics from options data."""
        if options_df is None or options_df.empty:
            return {}
        
        try:
            calls = options_df[options_df['type'] == 'call']
            puts = options_df[options_df['type'] == 'put']
            
            metrics = {
                'avg_iv': options_df['impliedVolatility'].mean(),
                'call_iv': calls['impliedVolatility'].mean(),
                'put_iv': puts['impliedVolatility'].mean(),
                'iv_skew': puts['impliedVolatility'].mean() - calls['impliedVolatility'].mean(),
                'put_call_ratio': puts['volume'].sum() / (calls['volume'].sum() + 1),
            }
            
            return metrics
        except:
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
            
        except:
            return {}