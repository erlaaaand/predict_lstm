"""
Export page component.
"""

import streamlit as st
import pandas as pd
from datetime import datetime


class ExportPage:
    """Renders export/download page."""
    
    @staticmethod
    def render(
        data: pd.DataFrame,
        ticker: str,
        prediction_results: dict = None,
        training_results: dict = None,
        model_config: dict = None
    ):
        """Render export page."""
        st.subheader("💾 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ExportPage._render_historical_export(data, ticker)
        
        with col2:
            if prediction_results:
                ExportPage._render_prediction_export(
                    prediction_results, ticker
                )
        
        with col3:
            if training_results and model_config:
                ExportPage._render_report_export(
                    data, ticker, prediction_results,
                    training_results, model_config
                )
        
        # Model architecture
        if training_results:
            ExportPage._render_model_architecture(training_results['model'])
    
    @staticmethod
    def _render_historical_export(data: pd.DataFrame, ticker: str):
        """Render historical data export."""
        st.write("**📊 Data Historis**")
        
        csv_data = data.to_csv()
        
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{ticker}_historical_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    @staticmethod
    def _render_prediction_export(results: dict, ticker: str):
        """Render prediction data export."""
        st.write("**🔮 Data Prediksi**")
        
        csv_data = results['dataframe'].to_csv()
        
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    @staticmethod
    def _render_report_export(
        data: pd.DataFrame,
        ticker: str,
        pred_results: dict,
        train_results: dict,
        config: dict
    ):
        """Render comprehensive report export."""
        st.write("**📈 Laporan Lengkap**")
        
        report = ExportPage._generate_report(
            data, ticker, pred_results, train_results, config
        )
        
        st.download_button(
            label="Download TXT",
            data=report,
            file_name=f"{ticker}_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    @staticmethod
    def _generate_report(
        data: pd.DataFrame,
        ticker: str,
        pred_results: dict,
        train_results: dict,
        config: dict
    ) -> str:
        """Generate comprehensive analysis report."""
        stats = pred_results['stats'] if pred_results else {}
        metrics = train_results['metrics'] if train_results else {}
        
        report = f"""
================================================================================
                    LAPORAN ANALISIS PREDIKSI SAHAM
================================================================================

INFORMASI UMUM
--------------
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {ticker}

DATA HISTORIS
-------------
Periode: {data.index[0].date()} s/d {data.index[-1].date()}
Jumlah Data: {len(data)} hari perdagangan
Harga Terakhir: {data['Close'].iloc[-1]:,.2f}
Min Harga: {data['Close'].min():,.2f}
Max Harga: {data['Close'].max():,.2f}
Rata-rata: {data['Close'].mean():,.2f}
Volatilitas: {data['Close'].std():,.2f}

KONFIGURASI MODEL
-----------------
Model Type: {config.get('model_type', 'N/A')}
Lookback Period: {config.get('lookback', 'N/A')} hari
LSTM Units (Layer 1): {config.get('lstm_units_1', 'N/A')}
LSTM Units (Layer 2): {config.get('lstm_units_2', 'N/A')}
Dropout Rate: {config.get('dropout', 'N/A')}
Learning Rate: {config.get('learning_rate', 'N/A')}
Epochs: {config.get('epochs', 'N/A')}
Batch Size: {config.get('batch_size', 'N/A')}
"""
        
        if metrics:
            report += f"""
PERFORMA MODEL
--------------
Training Set:
  - RMSE: {metrics['train']['rmse']:,.2f}
  - MAE: {metrics['train']['mae']:,.2f}
  - MAPE: {metrics['train']['mape']:.2f}%
  - R²: {metrics['train']['r2']:.4f}

Validation Set:
  - RMSE: {metrics['val']['rmse']:,.2f}
  - MAE: {metrics['val']['mae']:,.2f}
  - MAPE: {metrics['val']['mape']:.2f}%
  - R²: {metrics['val']['r2']:.4f}

Test Set:
  - RMSE: {metrics['test']['rmse']:,.2f}
  - MAE: {metrics['test']['mae']:,.2f}
  - MAPE: {metrics['test']['mape']:.2f}%
  - R²: {metrics['test']['r2']:.4f}
"""
        
        if stats:
            report += f"""
HASIL PREDIKSI
--------------
Harga Prediksi Akhir: {stats['final']:,.2f}
Expected Return: {stats['expected_return']:.2f}%
Min Prediksi: {stats['min']:,.2f}
Max Prediksi: {stats['max']:,.2f}
Rata-rata Prediksi: {stats['mean']:,.2f}
Standard Deviasi: {stats['std']:,.2f}
Range: {stats['range']:,.2f}

DISCLAIMER
----------
Prediksi ini hanya untuk tujuan edukasi dan penelitian.
Tidak boleh digunakan sebagai saran investasi.

================================================================================
"""
        
        return report
    
    @staticmethod
    def _render_model_architecture(model):
        """Render model architecture summary."""
        with st.expander("🏗️ Model Architecture"):
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
