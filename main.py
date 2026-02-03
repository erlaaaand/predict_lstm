"""
Quantitative Stock Analysis System
Advanced prediction with multi-source data and regularization.
FIXED VERSION - Comprehensive error handling and stability improvements.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.config.settings import config
from app.ui.styling import MinimalistTheme, render_header
from app.ui.sidebar import SidebarComponent
from app.ui.charts import ChartComponents
from app.services.analysis import AnalysisService
import warnings
import traceback

warnings.filterwarnings('ignore')


class QuantSystem:
    """Main application controller with robust error handling."""
    
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
        """Initialize session state with defaults."""
        defaults = {
            'analysis_complete': False,
            'data': None,
            'training_results': None,
            'predictions': None,
            'error_occurred': False,
            'error_message': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run application with error handling."""
        try:
            render_header(config.APP_TITLE)
            
            # Sidebar
            params = SidebarComponent.render()
            
            # Reset
            if params['reset']:
                self._reset()
                st.rerun()
            
            # Error display
            if st.session_state.get('error_occurred', False):
                st.error(f"❌ {st.session_state.get('error_message', 'Unknown error')}")
                if st.button("Clear Error"):
                    st.session_state.error_occurred = False
                    st.session_state.error_message = None
                    st.rerun()
            
            # Analysis
            if params['analyze']:
                self._run_analysis(params)
            
            # Display results
            if st.session_state.analysis_complete and not st.session_state.error_occurred:
                self._display_results()
            else:
                self._display_welcome()
                
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    def _reset(self):
        """Reset application state safely."""
        try:
            keys_to_keep = {'_is_running'}
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
        except Exception as e:
            st.error(f"Error during reset: {str(e)}")
    
    def _run_analysis(self, params: dict):
        """Run complete analysis pipeline with comprehensive error handling."""
        try:
            # Clear previous errors
            st.session_state.error_occurred = False
            st.session_state.error_message = None
            
            # Validate parameters
            if not self._validate_params(params):
                return
            
            # Step 1: Fetch data
            st.markdown("### 📊 Fetching Data")
            data_bundle = AnalysisService.fetch_all_data(
                params['ticker'],
                params['data_days']
            )
            
            if data_bundle is None:
                self._handle_error("Failed to fetch data. Please check ticker symbol.")
                return
            
            market_data = data_bundle['market_data']
            
            # Validate market data
            if market_data is None or len(market_data) < params['lookback'] + 50:
                self._handle_error(
                    f"Insufficient data: got {len(market_data) if market_data is not None else 0} rows, "
                    f"need at least {params['lookback'] + 50}"
                )
                return
            
            st.session_state.data = data_bundle
            
            # Step 2: Train model
            st.markdown("---")
            st.markdown("### 🤖 Training Model")
            
            training_results = AnalysisService.train_model(
                market_data,
                params,
                data_bundle['fundamental_data']
            )
            
            if training_results is None:
                self._handle_error("Model training failed.")
                return
            
            st.session_state.training_results = training_results
            
            # Display training summary
            test_metrics = training_results['metrics']['test']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test MAPE", f"{test_metrics['mape']:.2f}%")
            with col2:
                st.metric("Test R²", f"{test_metrics['r2']:.4f}")
            with col3:
                st.metric("Test MAE", f"{test_metrics['mae']:.4f}")
            with col4:
                st.metric("Test RMSE", f"{test_metrics['rmse']:.4f}")
            
            # Check model quality
            if test_metrics['mape'] > 20:
                st.warning("⚠️ Model MAPE > 20%. Consider adjusting parameters or using more data.")
            elif test_metrics['mape'] > 10:
                st.info("Model performance is acceptable. MAPE between 10-20%.")
            else:
                st.success("✓ Excellent model performance! MAPE < 10%")
            
            # Step 3: Make predictions
            st.markdown("---")
            st.markdown("### 🔮 Generating Predictions")
            
            predictions = AnalysisService.make_predictions(
                training_results['model'],
                market_data,
                training_results['prep_data'],
                params['predict_days'],
                training_results['is_ensemble']
            )
            
            if predictions is None:
                self._handle_error("Failed to generate predictions.")
                return
            
            st.session_state.predictions = predictions
            st.session_state.analysis_complete = True
            
            st.success("✅ Analysis complete! View results in the tabs below.")
            
            # Auto-scroll to results
            st.rerun()
            
        except Exception as e:
            self._handle_error(f"Analysis error: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    def _validate_params(self, params: dict) -> bool:
        """Validate analysis parameters."""
        try:
            # Ticker validation
            if not params.get('ticker') or not params['ticker'].strip():
                st.error("Please enter a valid ticker symbol")
                return False
            
            # Data days validation
            if params.get('data_days', 0) < 90:
                st.error("Minimum 90 days of data required")
                return False
            
            # Lookback validation
            if params.get('lookback', 0) < 10:
                st.error("Lookback period must be at least 10 days")
                return False
            
            if params.get('lookback', 0) > params.get('data_days', 0) / 2:
                st.error("Lookback period too large relative to data period")
                return False
            
            # Prediction days validation
            if params.get('predict_days', 0) < 1:
                st.error("Prediction period must be at least 1 day")
                return False
            
            if params.get('predict_days', 0) > 90:
                st.warning("Long prediction periods (>90 days) may be unreliable")
            
            return True
            
        except Exception as e:
            st.error(f"Parameter validation error: {str(e)}")
            return False
    
    def _handle_error(self, message: str):
        """Handle errors consistently."""
        st.session_state.error_occurred = True
        st.session_state.error_message = message
        st.session_state.analysis_complete = False
        st.error(f"❌ {message}")
    
    def _display_welcome(self):
        """Display welcome message with instructions."""
        st.markdown("""
        ### 👋 Welcome to Quantitative Stock Analysis System
        
        This system uses advanced machine learning techniques inspired by quantitative hedge funds.
        
        #### 🎯 Key Features:
        
        **Multi-Source Data:**
        - Market data (OHLCV)
        - 20+ technical indicators
        - Fundamental metrics (when available)
        - Options implied volatility (when available)
        - Analyst recommendations (when available)
        
        **Advanced ML Model:**
        - Regularized LSTM with dropout
        - Batch normalization
        - Early stopping & learning rate reduction
        - Optional ensemble models
        - Prevents overfitting
        
        **Robust Analysis:**
        - Comprehensive error handling
        - Data validation at every step
        - Multiple performance metrics
        - Confidence intervals on predictions
        
        #### 🚀 Getting Started:
        
        1. **Enter a ticker symbol** in the sidebar (e.g., AAPL, MSFT, BBCA.JK)
        2. **Configure parameters** (recommended: use defaults first)
        3. **Click "Run Analysis"** and wait 2-5 minutes
        4. **Explore results** in the tabs that appear below
        
        ---
        """)
        
        # Sample tickers
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📋 Sample US Tickers")
            us_samples = pd.DataFrame([
                {'Ticker': 'AAPL', 'Name': 'Apple Inc.', 'Sector': 'Technology'},
                {'Ticker': 'MSFT', 'Name': 'Microsoft Corp.', 'Sector': 'Technology'},
                {'Ticker': 'GOOGL', 'Name': 'Alphabet Inc.', 'Sector': 'Technology'},
                {'Ticker': 'JPM', 'Name': 'JPMorgan Chase', 'Sector': 'Finance'},
            ])
            st.dataframe(us_samples, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Sample Indonesian Tickers")
            id_samples = pd.DataFrame([
                {'Ticker': 'BBCA.JK', 'Name': 'Bank Central Asia', 'Sector': 'Banking'},
                {'Ticker': 'TLKM.JK', 'Name': 'Telkom Indonesia', 'Sector': 'Telecom'},
                {'Ticker': 'ASII.JK', 'Name': 'Astra International', 'Sector': 'Automotive'},
                {'Ticker': 'BBRI.JK', 'Name': 'Bank Rakyat Indonesia', 'Sector': 'Banking'},
            ])
            st.dataframe(id_samples, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Tips
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            **Data Selection:**
            - Use at least 365 days of historical data
            - More data = better model training
            - Check that the ticker has regular trading activity
            
            **Model Configuration:**
            - Start with default parameters
            - Bidirectional LSTM typically performs best
            - Ensemble mode improves stability (but takes longer)
            - Adjust dropout if you see overfitting (val_loss >> train_loss)
            
            **Interpreting Results:**
            - MAPE < 5%: Excellent
            - MAPE 5-10%: Good
            - MAPE 10-20%: Fair
            - MAPE > 20%: Consider retraining with different parameters
            
            **Performance Metrics:**
            - **MAPE**: Mean Absolute Percentage Error (lower is better)
            - **R²**: Coefficient of determination (closer to 1 is better)
            - **MAE**: Mean Absolute Error (lower is better)
            - **RMSE**: Root Mean Square Error (lower is better)
            
            **Predictions:**
            - Predictions are estimates, not guarantees
            - Always consider fundamental analysis
            - Use predictions as one tool among many
            - Never invest based solely on model predictions
            """)
        
        # Disclaimer
        st.warning("""
        ⚠️ **IMPORTANT DISCLAIMER:**
        
        This tool is for educational and research purposes only. Stock market predictions are inherently 
        uncertain and should not be used as the sole basis for investment decisions. Always conduct 
        thorough research and consult with qualified financial advisors before making investment decisions.
        """)
    
    def _display_results(self):
        """Display analysis results in organized tabs."""
        try:
            data = st.session_state.data
            training = st.session_state.training_results
            predictions = st.session_state.predictions
            
            if data is None or training is None or predictions is None:
                st.warning("Results not available. Please run analysis first.")
                return
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Market Data",
                "🤖 Model Training",
                "🔮 Predictions",
                "📈 Comprehensive Metrics"
            ])
            
            with tab1:
                self._display_market_data(data)
            
            with tab2:
                self._display_training(training)
            
            with tab3:
                self._display_predictions(predictions, data)
            
            with tab4:
                self._display_metrics(data, training, predictions)
                
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    def _display_market_data(self, data: dict):
        """Display market data analysis."""
        try:
            market_data = data['market_data']
            
            st.markdown("### Market Data Overview")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            last_price = market_data['Close'].iloc[-1]
            prev_price = market_data['Close'].iloc[-2] if len(market_data) > 1 else last_price
            change = last_price - prev_price
            change_pct = (change / prev_price * 100) if prev_price > 0 else 0
            
            with col1:
                st.metric("Current Price", f"{last_price:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
            with col2:
                st.metric("Period High", f"{market_data['High'].max():,.2f}")
            with col3:
                st.metric("Period Low", f"{market_data['Low'].min():,.2f}")
            with col4:
                volatility = market_data['Returns'].std() * np.sqrt(252) * 100 if 'Returns' in market_data.columns else 0
                st.metric("Annual Volatility", f"{volatility:.2f}%")
            
            # Price chart
            st.markdown("#### Price Chart")
            fig = ChartComponents.create_price_chart(market_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent data table
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Recent Prices")
                recent = market_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10)
                st.dataframe(
                    recent.style.format({
                        'Open': '{:,.2f}',
                        'High': '{:,.2f}',
                        'Low': '{:,.2f}',
                        'Close': '{:,.2f}',
                        'Volume': '{:,.0f}'
                    }),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### Key Statistics")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Std Dev', 'Min', 'Max', 'Data Points'],
                    'Value': [
                        f"{market_data['Close'].mean():,.2f}",
                        f"{market_data['Close'].std():,.2f}",
                        f"{market_data['Close'].min():,.2f}",
                        f"{market_data['Close'].max():,.2f}",
                        f"{len(market_data)}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            # Technical indicators
            if 'RSI' in market_data.columns:
                st.markdown("#### Technical Indicators")
                fig_rsi = ChartComponents.create_indicator_chart(
                    market_data.tail(60), 'RSI', 'Relative Strength Index (14)'
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error displaying market data: {str(e)}")
    
    def _display_training(self, training: dict):
        """Display training results."""
        try:
            st.markdown("### Training Results")
            
            # Training curve
            st.markdown("#### Loss Curves")
            fig = ChartComponents.create_training_chart(training['history'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics table
            st.markdown("#### Performance Metrics")
            metrics_df = pd.DataFrame({
                'Dataset': ['Train', 'Validation', 'Test'],
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
                }).background_gradient(subset=['R²'], cmap='RdYlGn'),
                hide_index=True,
                use_container_width=True
            )
            
            # Model info
            model_type = "Ensemble (3 models)" if training['is_ensemble'] else "Single Model"
            epochs_trained = len(training['history'].history['loss'])
            
            st.info(f"""
            **Model Configuration:**
            - Type: {model_type}
            - Epochs Trained: {epochs_trained}
            - Early Stopping: Enabled
            - Regularization: L1/L2 + Dropout + Batch Normalization
            """)
            
        except Exception as e:
            st.error(f"Error displaying training results: {str(e)}")
    
    def _display_predictions(self, predictions: dict, data: dict):
        """Display predictions."""
        try:
            market_data = data['market_data']
            
            st.markdown("### Future Price Predictions")
            
            # Summary metrics
            final_price = predictions['predictions'][-1]
            last_price = predictions['last_price']
            total_change = final_price - last_price
            total_change_pct = (total_change / last_price * 100) if last_price > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"{last_price:,.2f}")
            with col2:
                st.metric("Predicted Final Price", f"{final_price:,.2f}")
            with col3:
                st.metric("Expected Change", f"{total_change:+,.2f}", f"{total_change_pct:+.2f}%")
            
            # Prediction chart
            st.markdown("#### Prediction Chart")
            fig = ChartComponents.create_prediction_chart(
                market_data,
                predictions['predictions'],
                predictions['dates']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction table
            st.markdown("#### Daily Predictions")
            pred_df = pd.DataFrame({
                'Date': predictions['dates'],
                'Predicted Price': predictions['predictions']
            })
            
            # Calculate changes
            pred_df['Change'] = pred_df['Predicted Price'].diff()
            pred_df['Change %'] = pred_df['Predicted Price'].pct_change() * 100
            
            # Set first row
            pred_df.loc[0, 'Change'] = predictions['predictions'][0] - last_price
            pred_df.loc[0, 'Change %'] = ((predictions['predictions'][0] - last_price) / last_price * 100)
            
            st.dataframe(
                pred_df.style.format({
                    'Predicted Price': '{:,.2f}',
                    'Change': '{:+,.2f}',
                    'Change %': '{:+.2f}'
                }),
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Risk assessment
            st.markdown("#### Risk Assessment")
            
            expected_return = abs(total_change_pct)
            if expected_return < 5:
                risk_level = "🟢 Low Risk"
                risk_desc = "Expected change is relatively small"
            elif expected_return < 10:
                risk_level = "🟡 Medium Risk"
                risk_desc = "Moderate expected change"
            else:
                risk_level = "🔴 High Risk"
                risk_desc = "Significant expected change - high uncertainty"
            
            st.info(f"""
            **Risk Level:** {risk_level}
            
            {risk_desc}
            
            - Min Predicted: {predictions['predictions'].min():,.2f}
            - Max Predicted: {predictions['predictions'].max():,.2f}
            - Range: {predictions['predictions'].max() - predictions['predictions'].min():,.2f}
            - Std Dev: {predictions['predictions'].std():,.2f}
            """)
            
            # Disclaimer
            st.warning("""
            ⚠️ **IMPORTANT:** These predictions are statistical estimates based on historical data. 
            They should not be used as the sole basis for investment decisions. Always conduct 
            comprehensive research and consult financial advisors.
            """)
            
        except Exception as e:
            st.error(f"Error displaying predictions: {str(e)}")
    
    def _display_metrics(self, data: dict, training: dict, predictions: dict):
        """Display comprehensive metrics."""
        try:
            st.markdown("### Comprehensive Analysis")
            
            # Fundamental data
            if data.get('fundamental_data'):
                st.markdown("#### Fundamental Metrics")
                fund_df = pd.DataFrame([data['fundamental_data']]).T
                fund_df.columns = ['Value']
                fund_df = fund_df[fund_df['Value'] != 0]  # Remove zero values
                if not fund_df.empty:
                    st.dataframe(fund_df, use_container_width=True)
                else:
                    st.info("No fundamental data available")
            
            # IV metrics
            if data.get('iv_metrics'):
                st.markdown("#### Implied Volatility Metrics")
                iv_df = pd.DataFrame([data['iv_metrics']]).T
                iv_df.columns = ['Value']
                st.dataframe(iv_df.style.format('{:.4f}'), use_container_width=True)
            
            # Analyst data
            if data.get('analyst_data'):
                st.markdown("#### Analyst Recommendations")
                analyst_df = pd.DataFrame([data['analyst_data']]).T
                analyst_df.columns = ['Count']
                st.dataframe(analyst_df, use_container_width=True)
            
            # Model configuration
            st.markdown("#### Model Configuration")
            prep_data = training['prep_data']
            
            config_df = pd.DataFrame({
                'Parameter': [
                    'Input Features',
                    'Lookback Period',
                    'Training Samples',
                    'Validation Samples',
                    'Test Samples',
                    'Model Type'
                ],
                'Value': [
                    prep_data['n_features'],
                    prep_data['lookback'],
                    len(prep_data['X_train']),
                    len(prep_data['X_val']),
                    len(prep_data['X_test']),
                    'Ensemble' if training['is_ensemble'] else 'Single'
                ]
            })
            
            st.dataframe(config_df, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")


def main():
    """Application entry point with error handling."""
    try:
        app = QuantSystem()
        app.run()
    except Exception as e:
        st.error(f"Fatal application error: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        
        if st.button("Restart Application"):
            st.rerun()


if __name__ == "__main__":
    main()