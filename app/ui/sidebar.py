"""
Sidebar component for parameter configuration.
"""

import streamlit as st
from app.config import config


class SidebarComponent:
    """Renders sidebar with parameter controls."""
    
    @staticmethod
    def render() -> dict:
        """
        Render sidebar and return user configuration.
        
        Returns:
            Dictionary with user-selected parameters
        """
        with st.sidebar:
            st.header("🔧 Pengaturan Analisis")
            
            # Data parameters
            data_params = SidebarComponent._render_data_section()
            
            st.markdown("---")
            
            # Model parameters
            model_params = SidebarComponent._render_model_section()
            
            st.markdown("---")
            
            # Action buttons
            analyze_button = st.button(
                "🚀 Mulai Analisis",
                type="primary",
                use_container_width=True
            )
            
            reset_button = st.button(
                "🔄 Reset",
                use_container_width=True
            )
            
            return {
                **data_params,
                **model_params,
                'analyze_button': analyze_button,
                'reset_button': reset_button
            }
    
    @staticmethod
    def _render_data_section() -> dict:
        """Render data parameters section."""
        st.subheader("📊 Parameter Data")
        
        ticker = st.text_input(
            "Ticker Saham",
            value="BBCA.JK",
            help="Contoh: BBCA.JK, TLKM.JK, AAPL"
        ).upper()
        
        n_days_data = st.slider(
            "Jumlah Data (hari trading)",
            min_value=config.MIN_DATA_DAYS,
            max_value=config.MAX_DATA_DAYS,
            value=config.DEFAULT_DATA_DAYS,
            step=30
        )
        
        n_days_predict = st.slider(
            "Jumlah Hari Prediksi",
            min_value=config.MIN_PREDICT_DAYS,
            max_value=config.MAX_PREDICT_DAYS,
            value=config.DEFAULT_PREDICT_DAYS,
            step=1
        )
        
        return {
            'ticker': ticker,
            'n_days_data': n_days_data,
            'n_days_predict': n_days_predict
        }
    
    @staticmethod
    def _render_model_section() -> dict:
        """Render model parameters section."""
        st.subheader("🤖 Parameter Model")
        
        model_type = st.selectbox(
            "Tipe Model",
            config.MODEL_TYPES,
            index=1  # Default to Bidirectional
        )
        
        lookback = st.slider(
            "Lookback Period",
            min_value=config.MIN_LOOKBACK,
            max_value=config.MAX_LOOKBACK,
            value=config.DEFAULT_LOOKBACK,
            step=5
        )
        
        with st.expander("⚙️ Advanced Settings"):
            lstm_units_1 = st.slider(
                "LSTM Units (Layer 1)",
                min_value=config.MIN_LSTM_UNITS,
                max_value=config.MAX_LSTM_UNITS,
                value=config.DEFAULT_LSTM_UNITS_1,
                step=32
            )
            
            lstm_units_2 = st.slider(
                "LSTM Units (Layer 2)",
                min_value=32,
                max_value=128,
                value=config.DEFAULT_LSTM_UNITS_2,
                step=16
            )
            
            dropout = st.slider(
                "Dropout Rate",
                min_value=config.MIN_DROPOUT,
                max_value=config.MAX_DROPOUT,
                value=config.DEFAULT_DROPOUT,
                step=0.1
            )
            
            epochs = st.slider(
                "Epochs",
                min_value=config.MIN_EPOCHS,
                max_value=config.MAX_EPOCHS,
                value=config.DEFAULT_EPOCHS,
                step=25
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                config.BATCH_SIZES,
                index=1
            )
            
            learning_rate = st.select_slider(
                "Learning Rate",
                options=config.LEARNING_RATES,
                value=config.DEFAULT_LEARNING_RATE
            )
        
        return {
            'model_type': model_type,
            'lookback': lookback,
            'lstm_units_1': lstm_units_1,
            'lstm_units_2': lstm_units_2,
            'dropout': dropout,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
