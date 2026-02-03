"""
Technical analysis page component.
"""

import streamlit as st
import plotly.graph_objects as go
from app.ui.charts import ChartComponents


class TechnicalAnalysisPage:
    """Renders technical analysis page."""
    
    @staticmethod
    def render(data):
        """Render technical analysis page."""
        st.subheader("Analisis Teknikal")
        
        # Main candlestick chart
        fig = ChartComponents.create_candlestick_chart(data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional charts
        col1, col2 = st.columns(2)
        
        with col1:
            TechnicalAnalysisPage._render_returns_distribution(data)
        
        with col2:
            TechnicalAnalysisPage._render_volatility_chart(data)
    
    @staticmethod
    def _render_returns_distribution(data):
        """Render returns distribution histogram."""
        if 'Returns' not in data.columns:
            return
        
        st.subheader("Distribusi Returns")
        
        returns_data = data['Returns'].dropna() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns_data,
            nbinsx=50,
            name='Returns',
            marker_color='#3b82f6'
        ))
        
        # Hapus template fixed, biarkan Streamlit menangani warna
        fig.update_layout(
            xaxis_title='Returns (%)',
            yaxis_title='Frekuensi',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_volatility_chart(data):
        """Render rolling volatility chart."""
        if 'Volatility' not in data.columns:
            return
        
        st.subheader("Volatilitas Rolling (20 hari)")
        
        vol_data = data['Volatility'].dropna() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index[-len(vol_data):],
            y=vol_data,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#ef4444', width=2),
            name='Volatilitas'
        ))
        
        # Hapus template fixed
        fig.update_layout(
            xaxis_title='Tanggal',
            yaxis_title='Volatilitas (%)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)