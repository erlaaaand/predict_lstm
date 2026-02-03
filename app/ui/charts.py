"""
Minimalist chart components.
Simple, functional visualizations.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

class ChartComponents:
    """Simple chart components."""
    
    # Neutral color scheme
    COLORS = {
        'primary': '#18181b',
        'secondary': '#71717a',
        'border': '#e4e4e7',
        'up': '#22c55e',
        'down': '#ef4444'
    }
    
    @staticmethod
    def create_price_chart(data: pd.DataFrame) -> go.Figure:
        """Create simple price chart with moving averages."""
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#18181b',
            decreasing_line_color='#71717a'
        ))
        
        # Moving averages
        for ma in ['SMA_21', 'SMA_50']:
            if ma in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[ma],
                    name=ma,
                    line=dict(width=1),
                    opacity=0.7
                ))
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            margin=dict(l=10, r=10, t=30, b=10),
            font=dict(family='system-ui', size=12)
        )
        
        return fig
    
    @staticmethod
    def create_indicator_chart(data: pd.DataFrame, indicator: str, title: str) -> go.Figure:
        """Create simple indicator chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[indicator],
            line=dict(color='#18181b', width=1.5),
            name=indicator
        ))
        
        fig.update_layout(
            title=title,
            height=300,
            template='plotly_white',
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(family='system-ui', size=12)
        )
        
        return fig
    
    @staticmethod
    def create_training_chart(history) -> go.Figure:
        """Create training loss chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Train Loss',
            line=dict(color='#18181b', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Val Loss',
            line=dict(color='#71717a', width=2)
        ))
        
        fig.update_layout(
            title='Training Progress',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400,
            template='plotly_white',
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(family='system-ui', size=12)
        )
        
        return fig
    
    @staticmethod
    def create_prediction_chart(
        historical: pd.DataFrame,
        predictions: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> go.Figure:
        """Create prediction chart."""
        fig = go.Figure()
        
        # Historical
        hist_window = historical.tail(60)
        fig.add_trace(go.Scatter(
            x=hist_window.index,
            y=hist_window['Close'],
            name='Historical',
            line=dict(color='#18181b', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            name='Prediction',
            line=dict(color='#71717a', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=5)
        ))
        
        # Separator line
        if len(historical) > 0:
            fig.add_vline(
                x=historical.index[-1],
                line_dash="dot",
                line_color='#e4e4e7'
            )
        
        fig.update_layout(
            title='Price Prediction',
            height=500,
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(family='system-ui', size=12)
        )
        
        return fig