"""
Landing page component.
"""

import streamlit as st
import pandas as pd
from app.config import config


class LandingPage:
    """Renders landing/welcome page."""
    
    @staticmethod
    def render():
        """Render landing page."""
        st.info("""
        ### 👋 Selamat Datang di Aplikasi Prediksi Saham
        
        **Mulai dengan mengatur parameter di sidebar:**
        
        1. 📊 Pilih ticker saham yang ingin dianalisis
        2. ⚙️ Tentukan jumlah data historis yang akan digunakan
        3. 🤖 Atur parameter model LSTM sesuai kebutuhan
        4. 🚀 Klik tombol "Mulai Analisis" untuk memulai
        """)
        
        # Features
        LandingPage._render_features()
        
        # Sample tickers
        LandingPage._render_sample_tickers()
        
        # Tips
        LandingPage._render_tips()
    
    @staticmethod
    def _render_features():
        """Render feature list."""
        st.subheader("✨ Fitur Aplikasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - 📊 Analisis data historis dan teknikal
            - 🤖 Multiple LSTM architectures
            - 🔮 Prediksi harga masa depan
            """)
        
        with col2:
            st.markdown("""
            - 📈 Visualisasi interaktif
            - 💾 Export data dan laporan
            - ⚡ Caching untuk performa optimal
            """)
    
    @staticmethod
    def _render_sample_tickers():
        """Render sample tickers table."""
        st.subheader("🏢 Contoh Ticker Saham")
        
        sample_data = []
        for sector, info in config.SAMPLE_TICKERS.items():
            sample_data.append({
                'Sektor': sector,
                'Ticker': info['ticker'],
                'Nama': info['name']
            })
        
        df = pd.DataFrame(sample_data)
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def _render_tips():
        """Render usage tips."""
        with st.expander("💡 Tips Penggunaan"):
            st.markdown("""
            **Untuk Hasil Terbaik:**
            
            - Gunakan minimal 365 hari data untuk training yang lebih baik
            - Bidirectional LSTM umumnya memberikan hasil lebih akurat
            - Perhatikan metrik MAPE dan R² untuk menilai akurasi model
            - Gunakan Early Stopping untuk mencegah overfitting
            - Volatilitas tinggi memerlukan lebih banyak data training
            
            **Parameter yang Direkomendasikan:**
            
            - Lookback: 60 hari
            - LSTM Units Layer 1: 128
            - LSTM Units Layer 2: 64
            - Dropout: 0.2
            - Learning Rate: 0.001
            - Epochs: 100 (dengan early stopping)
            """)
