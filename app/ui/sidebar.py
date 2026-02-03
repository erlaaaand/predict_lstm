"""
Minimalist sidebar component.
Simple, functional, no clutter.
"""

import streamlit as st
from config.settings import config

class SidebarComponent:
    """Clean sidebar for parameter configuration."""
    
    @staticmethod
    def render() -> dict:
        """Render sidebar and return configuration."""
        with st.sidebar:
            st.markdown("### Configuration")
            
            # Data section
            params = SidebarComponent._render_data_params()
            
            st.markdown("---")
            
            # Model section
            model_params = SidebarComponent._render_model_params()
            params.update(model_params)
            
            st.markdown("---")
            
            # Action buttons
            analyze = st.button("Run Analysis", type="primary", use_container_width=True)
            reset = st.button("Reset", use_container_width=True)
            
            params['analyze'] = analyze
            params['reset'] = reset
            
            return params
    
    @staticmethod
    def _render_data_params() -> dict:
        """Render data configuration."""
        st.markdown("#### Data")
        
        ticker = st.text_input(
            "Ticker",
            value="BBCA.JK",
            help="Stock ticker symbol"
        ).upper()
        
        data_days = st.slider(
            "Data Period (days)",
            min_value=config.MIN_DATA_DAYS,
            max_value=config.MAX_DATA_DAYS,
            value=config.DEFAULT_DATA_DAYS,
            step=90
        )
        
        predict_days = st.slider(
            "Prediction Period (days)",
            min_value=1,
            max_value=60,
            value=14,
            step=1
        )
        
        return {
            'ticker': ticker,
            'data_days': data_days,
            'predict_days': predict_days
        }
    
    @staticmethod
    def _render_model_params() -> dict:
        """Render model configuration."""
        st.markdown("#### Model")
        
        lookback = st.slider(
            "Lookback Period",
            min_value=config.MIN_LOOKBACK,
            max_value=config.MAX_LOOKBACK,
            value=config.DEFAULT_LOOKBACK,
            step=10
        )
        
        use_ensemble = st.checkbox("Use Ensemble", value=True)
        
        with st.expander("Advanced"):
            units_1 = st.slider("LSTM Units L1", 64, 256, 128, 32)
            units_2 = st.slider("LSTM Units L2", 32, 128, 64, 16)
            dropout = st.slider("Dropout", 0.2, 0.5, 0.3, 0.05)
            epochs = st.slider("Epochs", 50, 200, 100, 25)
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005],
                value=0.001
            )
        
        return {
            'lookback': lookback,
            'use_ensemble': use_ensemble,
            'units_1': units_1,
            'units_2': units_2,
            'dropout': dropout,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }