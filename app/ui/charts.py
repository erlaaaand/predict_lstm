"""
Visualization components using Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional


class ChartComponents:
    """Reusable chart components."""
    
    @staticmethod
    def create_candlestick_chart(
        data: pd.DataFrame,
        height: int = 800
    ) -> go.Figure:
        """
        Create candlestick chart with technical indicators.
        
        Args:
            data: Stock data with OHLCV
            height: Chart height
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                'Harga Saham dengan Moving Averages',
                'Volume Perdagangan',
                'RSI'
            ),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Moving Averages
        for ma, color in [('MA7', 'orange'), ('MA21', 'blue'), ('MA50', 'red')]:
            if ma in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[ma],
                        name=ma,
                        line=dict(color=color, width=1)
                    ),
                    row=1, col=1
                )
        
        # Volume
        colors = [
            'red' if row['Close'] < row['Open'] else 'green'
            for _, row in data.iterrows()
        ]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
    
    @staticmethod
    def create_training_curves(history) -> tuple:
        """
        Create training loss and MAE curves.
        
        Args:
            history: Keras training history
            
        Returns:
            Tuple of (loss_fig, mae_fig)
        """
        # Loss curve
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training Loss',
            mode='lines',
            line=dict(color='#3b82f6', width=2)
        ))
        loss_fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Validation Loss',
            mode='lines',
            line=dict(color='#ef4444', width=2)
        ))
        loss_fig.update_layout(
            title='Model Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            height=400,
            template='plotly_white'
        )
        
        # MAE curve
        mae_fig = go.Figure()
        if 'mae' in history.history:
            mae_fig.add_trace(go.Scatter(
                y=history.history['mae'],
                name='Training MAE',
                mode='lines',
                line=dict(color='#22c55e', width=2)
            ))
        if 'val_mae' in history.history:
            mae_fig.add_trace(go.Scatter(
                y=history.history['val_mae'],
                name='Validation MAE',
                mode='lines',
                line=dict(color='#f59e0b', width=2)
            ))
        mae_fig.update_layout(
            title='Mean Absolute Error',
            xaxis_title='Epoch',
            yaxis_title='MAE',
            height=400,
            template='plotly_white'
        )
        
        return loss_fig, mae_fig
    
    @staticmethod
    def create_prediction_chart(
        historical_data: pd.DataFrame,
        train_pred: Optional[np.ndarray] = None,
        val_pred: Optional[np.ndarray] = None,
        test_pred: Optional[np.ndarray] = None,
        train_dates=None,
        val_dates=None,
        test_dates=None
    ) -> go.Figure:
        """Create prediction vs actual chart."""
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='Actual',
            mode='lines',
            line=dict(color='black', width=1.5)
        ))
        
        # Predictions
        if train_pred is not None and train_dates is not None:
            fig.add_trace(go.Scatter(
                x=train_dates,
                y=train_pred,
                name='Train Predictions',
                mode='lines',
                line=dict(color='#3b82f6', width=2)
            ))
        
        if val_pred is not None and val_dates is not None:
            fig.add_trace(go.Scatter(
                x=val_dates,
                y=val_pred,
                name='Val Predictions',
                mode='lines',
                line=dict(color='#f59e0b', width=2)
            ))
        
        if test_pred is not None and test_dates is not None:
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_pred,
                name='Test Predictions',
                mode='lines',
                line=dict(color='#ef4444', width=2)
            ))
        
        fig.update_layout(
            title='Prediksi Model vs Harga Aktual',
            xaxis_title='Tanggal',
            yaxis_title='Harga',
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_future_prediction_chart(
        historical_data: pd.DataFrame,
        predictions: np.ndarray,
        future_dates: pd.DatetimeIndex,
        historical_window: int = 60
    ) -> go.Figure:
        """Create future prediction chart with confidence interval."""
        fig = go.Figure()
        
        # Historical data (last N days)
        hist_slice = historical_data.tail(historical_window)
        fig.add_trace(go.Scatter(
            x=hist_slice.index,
            y=hist_slice['Close'],
            name='Historical',
            mode='lines',
            line=dict(color='#18181b', width=2)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            name='Predictions',
            mode='lines+markers',
            line=dict(color='#3b82f6', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Confidence interval (simplified)
        pred_std = predictions.std()
        upper_bound = predictions + 2 * pred_std
        lower_bound = predictions - 2 * pred_std
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(59, 130, 246, 0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(59, 130, 246, 0)',
            name='95% Confidence',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        # Add vertical line
        if len(historical_data) > 0:
            fig.add_vline(
                x=historical_data.index[-1],
                line_dash="dot",
                line_color="gray",
                annotation_text="Prediction Start"
            )
        
        fig.update_layout(
            title='Prediksi Harga Masa Depan',
            xaxis_title='Tanggal',
            yaxis_title='Harga',
            height=600,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
