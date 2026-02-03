"""
Quantitative Stock Analysis System
Advanced prediction with multi-source data and regularization.
"""

import streamlit as st
import pandas as pd
from config.settings import config
from app.ui.styling import MinimalistTheme, render_header
from app.ui.sidebar import SidebarComponent
from app.ui.charts import ChartComponents
from app.services.analysis import AnalysisService

class QuantSystem:
    """Main application controller."""
    
    def __init__(self):
        self._configure_page()
        self._init_session_state()
    
    def _configure_page(self):
        """Configure page settings."""
        st.set_page_config(
            page_title=config.APP_TITLE,
            page_icon=config.APP_ICON,
            layout=config.LAYOUT,
            initial_sidebar_state="expanded"
        )
        MinimalistTheme.apply()
    
    def _init_session_state(self):
        """Initialize session state."""
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'training_results' not in st.session_state:
            st.session_state.training_results = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
    
    def run(self):
        """Run application."""
        render_header(config.APP_TITLE)
        
        # Sidebar
        params = SidebarComponent.render()
        
        # Reset
        if params['reset']:
            self._reset()
            st.rerun()
        
        # Analysis
        if params['analyze']:
            self._run_analysis(params)
        
        # Display results
        if st.session_state.analysis_complete:
            self._display_results()
        else:
            self._display_welcome()
    
    def _reset(self):
        """Reset application state."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    def _run_analysis(self, params: dict):
        """Run complete analysis pipeline."""
        
        # Step 1: Fetch data
        data_bundle = AnalysisService.fetch_all_data(
            params['ticker'],
            params['data_days']
        )
        
        if data_bundle is None:
            st.error("Failed to fetch data")
            return
        
        st.session_state.data = data_bundle
        
        market_data = data_bundle['market_data']
        fundamental_data = data_bundle['fundamental_data']
        
        st.success(f"Fetched {len(market_data)} days of data")
        
        # Step 2: Train model
        training_results = AnalysisService.train_model(
            market_data,
            params,
            fundamental_data
        )
        
        if training_results is None:
            st.error("Training failed")
            return
        
        st.session_state.training_results = training_results
        
        # Display training metrics
        test_metrics = training_results['metrics']['test']
        st.success(
            f"Model trained - Test MAPE: {test_metrics['mape']:.2f}%, "
            f"R²: {test_metrics['r2']:.4f}"
        )
        
        # Step 3: Make predictions
        predictions = AnalysisService.make_predictions(
            training_results['model'],
            market_data,
            training_results['prep_data'],
            params['predict_days'],
            training_results['is_ensemble']
        )
        
        st.session_state.predictions = predictions
        st.session_state.analysis_complete = True
        
        st.rerun()
    
    def _display_welcome(self):
        """Display welcome message."""
        st.markdown("""
        ### Welcome to Quantitative Stock Analysis System
        
        This system uses advanced techniques inspired by quantitative hedge funds:
        
        **Data Sources:**
        - Market data (OHLCV)
        - Technical indicators (20+ indicators)
        - Fundamental metrics
        - Options implied volatility
        - Analyst recommendations
        
        **Model Features:**
        - Regularized LSTM with dropout
        - Batch normalization
        - Early stopping
        - Optional ensemble models
        - Prevents overfitting
        
        Configure parameters in the sidebar and click "Run Analysis" to start.
        """)
        
        # Sample tickers
        st.markdown("#### Sample Tickers")
        samples = pd.DataFrame([
            {'Ticker': 'BBCA.JK', 'Name': 'Bank Central Asia'},
            {'Ticker': 'TLKM.JK', 'Name': 'Telkom Indonesia'},
            {'Ticker': 'AAPL', 'Name': 'Apple Inc.'},
            {'Ticker': 'MSFT', 'Name': 'Microsoft Corp.'},
        ])
        st.dataframe(samples, hide_index=True, use_container_width=True)
    
    def _display_results(self):
        """Display analysis results."""
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Market Data",
            "Model Training",
            "Predictions",
            "Metrics"
        ])
        
        data = st.session_state.data
        training = st.session_state.training_results
        predictions = st.session_state.predictions
        
        with tab1:
            self._display_market_data(data)
        
        with tab2:
            self._display_training(training)
        
        with tab3:
            self._display_predictions(predictions, data)
        
        with tab4:
            self._display_metrics(data, training)
    
    def _display_market_data(self, data: dict):
        """Display market data."""
        market_data = data['market_data']
        
        st.markdown("### Market Data")
        
        # Price chart
        fig = ChartComponents.create_price_chart(market_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent data
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Recent Prices")
            recent = market_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
            st.dataframe(recent, use_container_width=True)
        
        with col2:
            st.markdown("#### Key Indicators")
            if 'RSI' in market_data.columns:
                fig_rsi = ChartComponents.create_indicator_chart(
                    market_data.tail(60), 'RSI', 'RSI (14)'
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
    
    def _display_training(self, training: dict):
        """Display training results."""
        st.markdown("### Training Results")
        
        # Training curve
        fig = ChartComponents.create_training_chart(training['history'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table
        st.markdown("#### Performance Metrics")
        metrics_df = pd.DataFrame({
            'Split': ['Train', 'Validation', 'Test'],
            'MAE': [
                training['metrics']['train']['mae'],
                training['metrics']['val']['mae'],
                training['metrics']['test']['mae']
            ],
            'RMSE': [
                training['metrics']['train']['rmse'],
                training['metrics']['val']['rmse'],
                training['metrics']['test']['rmse']
            ],
            'MAPE (%)': [
                training['metrics']['train']['mape'],
                training['metrics']['val']['mape'],
                training['metrics']['test']['mape']
            ],
            'R²': [
                training['metrics']['train']['r2'],
                training['metrics']['val']['r2'],
                training['metrics']['test']['r2']
            ]
        })
        
        st.dataframe(
            metrics_df.style.format({
                'MAE': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAPE (%)': '{:.2f}',
                'R²': '{:.4f}'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    def _display_predictions(self, predictions: dict, data: dict):
        """Display predictions."""
        st.markdown("### Future Predictions")
        
        market_data = data['market_data']
        
        # Prediction chart
        fig = ChartComponents.create_prediction_chart(
            market_data,
            predictions['predictions'],
            predictions['dates']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction table
        st.markdown("#### Predicted Prices")
        pred_df = pd.DataFrame({
            'Date': predictions['dates'],
            'Predicted Price': predictions['predictions']
        })
        
        # Calculate change
        pred_df['Change'] = pred_df['Predicted Price'].diff()
        pred_df['Change %'] = pred_df['Predicted Price'].pct_change() * 100
        
        st.dataframe(
            pred_df.style.format({
                'Predicted Price': '{:,.2f}',
                'Change': '{:+,.2f}',
                'Change %': '{:+.2f}'
            }),
            hide_index=True,
            use_container_width=True
        )
        
        # Summary
        last_price = predictions['last_price']
        final_price = predictions['predictions'][-1]
        total_change = ((final_price - last_price) / last_price) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"{last_price:,.2f}")
        with col2:
            st.metric("Predicted Price", f"{final_price:,.2f}")
        with col3:
            st.metric("Expected Change", f"{total_change:+.2f}%")
    
    def _display_metrics(self, data: dict, training: dict):
        """Display comprehensive metrics."""
        st.markdown("### Comprehensive Metrics")
        
        # Fundamental data
        if data['fundamental_data']:
            st.markdown("#### Fundamental Metrics")
            fund_df = pd.DataFrame([data['fundamental_data']]).T
            fund_df.columns = ['Value']
            st.dataframe(fund_df, use_container_width=True)
        
        # IV metrics
        if data['iv_metrics']:
            st.markdown("#### Implied Volatility Metrics")
            iv_df = pd.DataFrame([data['iv_metrics']]).T
            iv_df.columns = ['Value']
            st.dataframe(iv_df, use_container_width=True)
        
        # Analyst data
        if data['analyst_data']:
            st.markdown("#### Analyst Recommendations")
            analyst_df = pd.DataFrame([data['analyst_data']]).T
            analyst_df.columns = ['Count']
            st.dataframe(analyst_df, use_container_width=True)
        
        # Model info
        st.markdown("#### Model Configuration")
        st.info("""
        **Regularization Techniques Applied:**
        - Dropout layers (prevent co-adaptation)
        - Recurrent dropout (stabilize training)
        - L1/L2 regularization (weight penalty)
        - Batch normalization (normalize activations)
        - Early stopping (prevent overtraining)
        - Learning rate reduction (fine-tuning)
        
        **Ensemble:** {} model(s)
        """.format("3" if training['is_ensemble'] else "1"))


def main():
    """Entry point."""
    app = QuantSystem()
    app.run()


if __name__ == "__main__":
    main()