"""
Complete chart components with all methods.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np


class ChartComponents:
    """Chart components for visualization."""
    
    COLORS = {
        'primary': '#18181b',
        'secondary': '#71717a',
        'border': '#e4e4e7',
        'up': '#22c55e',
        'down': '#ef4444'
    }
    
    @staticmethod
    def create_price_chart(data: pd.DataFrame) -> go.Figure:
        """Create price chart with moving averages."""
        fig = go.Figure()
        
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
    def create_candlestick_chart(data: pd.DataFrame) -> go.Figure:
        """Create candlestick chart with volume."""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price', 'Volume', 'RSI')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        for ma in ['SMA_7', 'SMA_21', 'SMA_50']:
            if ma in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data[ma], name=ma, line=dict(width=1)),
                    row=1, col=1
                )
        
        # Volume
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                  for _, row in data.iterrows()]
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple', width=1)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=800,
            template='plotly_white',
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @staticmethod
    def create_indicator_chart(data: pd.DataFrame, indicator: str, title: str) -> go.Figure:
        """Create indicator chart."""
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
    def create_training_curves(history):
        """Create training curves for loss and MAE."""
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
        loss_fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'))
        loss_fig.update_layout(title='Loss', template='plotly_white', height=350)
        
        mae_fig = go.Figure()
        if 'mae' in history.history:
            mae_fig.add_trace(go.Scatter(y=history.history['mae'], name='Train MAE'))
            mae_fig.add_trace(go.Scatter(y=history.history['val_mae'], name='Val MAE'))
        mae_fig.update_layout(title='MAE', template='plotly_white', height=350)
        
        return loss_fig, mae_fig
    
    @staticmethod
    def create_prediction_chart(
        historical: pd.DataFrame,
        predictions: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> go.Figure:
        """Create prediction chart."""
        fig = go.Figure()
        
        hist_window = historical.tail(60)
        fig.add_trace(go.Scatter(
            x=hist_window.index,
            y=hist_window['Close'],
            name='Historical',
            line=dict(color='#18181b', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            name='Prediction',
            line=dict(color='#71717a', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=5)
        ))
        
        if len(historical) > 0:
            fig.add_vline(x=historical.index[-1], line_dash="dot", line_color='#e4e4e7')
        
        fig.update_layout(
            title='Price Prediction',
            height=500,
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(family='system-ui', size=12)
        )
        
        return fig
    
    @staticmethod
    def create_future_prediction_chart(
        historical: pd.DataFrame,
        predictions: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> go.Figure:
        """Create future prediction chart with confidence interval."""
        return ChartComponents.create_prediction_chart(historical, predictions, dates)