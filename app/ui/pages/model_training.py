"""
Model training page component.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app.ui.charts import ChartComponents


class ModelTrainingPage:
    """Renders model training page."""
    
    @staticmethod
    def render(training_results: dict, data, lookback: int):
        """Render model training results page."""
        st.subheader("Hasil Pelatihan Model")
        
        # Training curves
        ModelTrainingPage._render_training_curves(training_results['history'])
        
        # Metrics table
        ModelTrainingPage._render_metrics_table(training_results['metrics'])
        
        # Prediction visualization
        ModelTrainingPage._render_predictions(
            training_results, data, lookback
        )
        
        # Residual analysis
        ModelTrainingPage._render_residual_analysis(training_results)
        
        # Model info
        ModelTrainingPage._render_model_info(training_results)
    
    @staticmethod
    def _render_training_curves(history):
        """Render training loss and MAE curves."""
        st.subheader("Kurva Pelatihan")
        
        col1, col2 = st.columns(2)
        
        loss_fig, mae_fig = ChartComponents.create_training_curves(history)
        
        with col1:
            st.plotly_chart(loss_fig, use_container_width=True)
        
        with col2:
            st.plotly_chart(mae_fig, use_container_width=True)
    
    @staticmethod
    def _render_metrics_table(metrics: dict):
        """Render metrics comparison table."""
        st.subheader("Metrik Evaluasi")
        
        metrics_df = pd.DataFrame({
            'Dataset': ['Training', 'Validation', 'Test'],
            'RMSE': [
                metrics['train']['rmse'],
                metrics['val']['rmse'],
                metrics['test']['rmse']
            ],
            'MAE': [
                metrics['train']['mae'],
                metrics['val']['mae'],
                metrics['test']['mae']
            ],
            'MAPE (%)': [
                metrics['train']['mape'],
                metrics['val']['mape'],
                metrics['test']['mape']
            ],
            'R²': [
                metrics['train']['r2'],
                metrics['val']['r2'],
                metrics['test']['r2']
            ]
        })
        
        st.dataframe(
            metrics_df.style.format({
                'RMSE': '{:,.2f}',
                'MAE': '{:,.2f}',
                'MAPE (%)': '{:.2f}',
                'R²': '{:.4f}'
            }).background_gradient(subset=['R²'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    @staticmethod
    def _render_predictions(results: dict, data, lookback: int):
        """Render prediction vs actual chart."""
        st.subheader("Prediksi vs Aktual")
        
        model = results['model']
        scaler = results['scaler']
        prep_data = results['prepared_data']
        
        # Make predictions
        train_pred = model.predict(prep_data['X_train'], verbose=0)
        val_pred = model.predict(prep_data['X_val'], verbose=0)
        test_pred = model.predict(prep_data['X_test'], verbose=0)
        
        # Inverse transform
        train_pred = scaler.inverse_transform(train_pred).flatten()
        val_pred = scaler.inverse_transform(val_pred).flatten()
        test_pred = scaler.inverse_transform(test_pred).flatten()
        
        # Generate dates
        train_size = len(prep_data['X_train'])
        val_size = len(prep_data['X_val'])
        
        train_dates = data.index[lookback:train_size + lookback]
        val_dates = data.index[train_size + lookback:train_size + val_size + lookback]
        test_dates = data.index[train_size + val_size + lookback:]
        
        # Create chart
        fig = ChartComponents.create_prediction_chart(
            data, train_pred, val_pred, test_pred,
            train_dates, val_dates, test_dates
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_residual_analysis(results: dict):
        """Render residual analysis."""
        st.subheader("Analisis Residual (Test Set)")
        
        model = results['model']
        scaler = results['scaler']
        prep_data = results['prepared_data']
        
        # Get test predictions
        test_pred = model.predict(prep_data['X_test'], verbose=0)
        test_pred = scaler.inverse_transform(test_pred).flatten()
        test_actual = scaler.inverse_transform(
            prep_data['y_test'].reshape(-1, 1)
        ).flatten()
        
        residuals = test_actual - test_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residual scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=test_pred,
                y=residuals,
                mode='markers',
                marker=dict(color='#6366f1', size=5),
                name='Residuals'
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            # Hapus template='plotly_white'
            fig.update_layout(
                title='Residual Plot',
                xaxis_title='Predicted Values',
                yaxis_title='Residuals',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residual histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='#14b8a6',
                name='Residuals'
            ))
            
            # Hapus template='plotly_white'
            fig.update_layout(
                title='Distribusi Residual',
                xaxis_title='Residual',
                yaxis_title='Frekuensi',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_model_info(results: dict):
        """Render model information."""
        model = results['model']
        metrics = results['metrics']
        
        # Mengganti st.success emoji dengan format bersih
        st.success(f"""
        **Model Berhasil Dilatih**
        
        - Total Parameters: {model.count_params():,}
        - Test MAPE: {metrics['test']['mape']:.2f}%
        - Test R²: {metrics['test']['r2']:.4f}
        """)