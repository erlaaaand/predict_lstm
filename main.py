"""
Stock Price Prediction Application - Main Entry Point
"""

import streamlit as st
from app.config import config
from app.services import StockAnalysisService
from app.ui import UITheme, SidebarComponent, render_header
from app.ui.pages import (
    LandingPage,
    DataStatsPage,
    TechnicalAnalysisPage,
    ModelTrainingPage,
    PredictionPage,
    ExportPage
)


class StockPredictionApp:
    """Main application controller."""
    
    def __init__(self):
        """Initialize application."""
        self._configure_page()
        self._initialize_session_state()
    
    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=config.APP_TITLE,
            page_icon=config.APP_ICON,
            layout=config.LAYOUT,
            initial_sidebar_state=config.SIDEBAR_STATE
        )
        
        # Apply custom styling
        UITheme.apply()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'stock_data' not in st.session_state:
            st.session_state.stock_data = None
        if 'training_results' not in st.session_state:
            st.session_state.training_results = None
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = None
        if 'user_config' not in st.session_state:
            st.session_state.user_config = None
    
    def run(self):
        """Run the application."""
        # Render header
        render_header(config.APP_TITLE, config.APP_ICON)
        
        # Render sidebar and get user configuration
        user_config = SidebarComponent.render()
        
        # Handle reset button
        if user_config['reset_button']:
            self._reset_application()
            st.rerun()
        
        # Handle analyze button
        if user_config['analyze_button']:
            self._run_analysis(user_config)
        
        # Render appropriate page
        if not st.session_state.model_trained:
            LandingPage.render()
        else:
            self._render_results_tabs()
    
    def _reset_application(self):
        """Reset all session state."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    def _run_analysis(self, user_config: dict):
        """
        Run complete stock analysis pipeline.
        
        Args:
            user_config: User configuration from sidebar
        """
        # Extract configuration
        ticker = user_config['ticker']
        n_days_data = user_config['n_days_data']
        n_days_predict = user_config['n_days_predict']
        
        # Step 1: Fetch and prepare data
        st.info("📊 Mengambil dan memproses data saham...")
        stock_data = StockAnalysisService.fetch_and_prepare_data(
            ticker, n_days_data
        )
        
        if stock_data is None:
            st.stop()
        
        st.success(
            f"✅ Berhasil mengambil {len(stock_data)} hari perdagangan "
            f"dari {stock_data.index[0].date()} hingga {stock_data.index[-1].date()}"
        )
        
        # Store in session state
        st.session_state.stock_data = stock_data
        st.session_state.user_config = user_config
        
        # Step 2: Train model
        st.info("🤖 Melatih model LSTM...")
        
        model_config = {
            'lookback': user_config['lookback'],
            'lstm_units_1': user_config['lstm_units_1'],
            'lstm_units_2': user_config['lstm_units_2'],
            'dropout': user_config['dropout'],
            'model_type': user_config['model_type'],
            'learning_rate': user_config['learning_rate'],
            'epochs': user_config['epochs'],
            'batch_size': user_config['batch_size']
        }
        
        training_results = StockAnalysisService.train_model(
            stock_data, model_config
        )
        
        if training_results is None:
            st.stop()
        
        st.session_state.training_results = training_results
        
        # Step 3: Make predictions
        st.info(f"🔮 Membuat prediksi untuk {n_days_predict} hari ke depan...")
        
        prediction_results = StockAnalysisService.make_predictions(
            model=training_results['model'],
            data=stock_data,
            scaler=training_results['scaler'],
            lookback=user_config['lookback'],
            n_days=n_days_predict
        )
        
        if prediction_results is None:
            st.error("Gagal membuat prediksi")
            st.stop()
        
        st.session_state.prediction_results = prediction_results
        st.session_state.model_trained = True
        
        st.success("✅ Analisis selesai! Jelajahi hasil di tab-tab berikut.")
        st.rerun()
    
    def _render_results_tabs(self):
        """Render results in tabs."""
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data & Statistik",
            "📈 Analisis Teknikal",
            "🤖 Model LSTM",
            "🔮 Prediksi",
            "💾 Export"
        ])
        
        data = st.session_state.stock_data
        training_results = st.session_state.training_results
        prediction_results = st.session_state.prediction_results
        user_config = st.session_state.user_config
        
        with tab1:
            DataStatsPage.render(data)
        
        with tab2:
            TechnicalAnalysisPage.render(data)
        
        with tab3:
            ModelTrainingPage.render(
                training_results,
                data,
                user_config['lookback']
            )
        
        with tab4:
            PredictionPage.render(
                prediction_results,
                data,
                user_config['ticker']
            )
        
        with tab5:
            ExportPage.render(
                data=data,
                ticker=user_config['ticker'],
                prediction_results=prediction_results,
                training_results=training_results,
                model_config=user_config
            )


def main():
    """Application entry point."""
    app = StockPredictionApp()
    app.run()


if __name__ == "__main__":
    main()
