"""
Data statistics page component.
"""

import streamlit as st
import pandas as pd


class DataStatsPage:
    """Renders data statistics page."""
    
    @staticmethod
    def render(data: pd.DataFrame):
        """Render data statistics page."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            DataStatsPage._render_recent_data(data)
        
        with col2:
            DataStatsPage._render_descriptive_stats(data)
        
        with col3:
            DataStatsPage._render_latest_metrics(data)
    
    @staticmethod
    def _render_recent_data(data: pd.DataFrame):
        """Render recent stock data table."""
        st.subheader("Data Terbaru")
        
        display_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15)
        
        st.dataframe(
            display_data.style.format({
                'Open': '{:,.2f}',
                'High': '{:,.2f}',
                'Low': '{:,.2f}',
                'Close': '{:,.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
    
    @staticmethod
    def _render_descriptive_stats(data: pd.DataFrame):
        """Render descriptive statistics."""
        st.subheader("Statistik Deskriptif")
        
        stats_df = pd.DataFrame({
            'Metrik': ['Min', 'Max', 'Mean', 'Median', 'Std Dev', 'Count'],
            'Nilai': [
                f"{data['Close'].min():,.2f}",
                f"{data['Close'].max():,.2f}",
                f"{data['Close'].mean():,.2f}",
                f"{data['Close'].median():,.2f}",
                f"{data['Close'].std():,.2f}",
                f"{len(data)}"
            ]
        })
        
        st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    @staticmethod
    def _render_latest_metrics(data: pd.DataFrame):
        """Render latest price metrics."""
        st.subheader("Harga Terakhir")
        
        last_close = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2] if len(data) > 1 else last_close
        change = last_close - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        st.metric(
            "Close Price",
            f"{last_close:,.2f}",
            f"{change:,.2f} ({change_pct:.2f}%)"
        )
        
        if 'Volume' in data.columns:
            vol_current = data['Volume'].iloc[-1]
            vol_prev = data['Volume'].iloc[-2] if len(data) > 1 else vol_current
            vol_change = ((vol_current / vol_prev - 1) * 100) if vol_prev != 0 else 0
            
            st.metric(
                "Volume",
                f"{vol_current:,.0f}",
                f"{vol_change:.2f}%"
            )
        
        if 'Returns' in data.columns:
            import numpy as np
            volatility = data['Returns'].std() * np.sqrt(252) * 100
            st.metric("Volatilitas Tahunan", f"{volatility:.2f}%")
