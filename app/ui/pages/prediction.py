"""
Prediction page component.
"""

import streamlit as st
import pandas as pd
from app.ui.charts import ChartComponents


class PredictionPage:
    """Renders prediction page."""
    
    @staticmethod
    def render(prediction_results: dict, data, ticker: str):
        """Render prediction results page."""
        st.subheader("Prediksi Harga Masa Depan")
        
        # Statistics cards
        PredictionPage._render_prediction_stats(prediction_results['stats'])
        
        # Prediction chart
        PredictionPage._render_prediction_chart(
            prediction_results, data, ticker
        )
        
        # Prediction table
        PredictionPage._render_prediction_table(prediction_results['dataframe'])
        
        # Risk assessment
        PredictionPage._render_risk_assessment(prediction_results['stats'])
        
        # Disclaimer
        PredictionPage._render_disclaimer()
    
    @staticmethod
    def _render_prediction_stats(stats: dict):
        """Render prediction statistics cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Harga Akhir Prediksi",
                f"{stats['final']:,.2f}",
                f"{stats['expected_return']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Rata-rata Prediksi",
                f"{stats['mean']:,.2f}"
            )
        
        with col3:
            st.metric(
                "Min Prediksi",
                f"{stats['min']:,.2f}"
            )
        
        with col4:
            st.metric(
                "Max Prediksi",
                f"{stats['max']:,.2f}"
            )
    
    @staticmethod
    def _render_prediction_chart(results: dict, data, ticker: str):
        """Render future prediction chart."""
        st.subheader("Visualisasi Prediksi")
        
        fig = ChartComponents.create_future_prediction_chart(
            data,
            results['predictions'],
            results['dates']
        )
        
        # Hapus template='plotly_white' agar adaptif terhadap Dark Mode
        fig.update_layout(title=f'Prediksi Harga {ticker}')
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_prediction_table(pred_df: pd.DataFrame):
        """Render prediction table."""
        st.subheader("Tabel Prediksi Harian")
        
        st.dataframe(
            pred_df.style.format({
                'Predicted_Price': '{:,.2f}',
                'Daily_Change': '{:+,.2f}',
                'Daily_Change_%': '{:+.2f}%'
            }).background_gradient(subset=['Predicted_Price'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    @staticmethod
    def _render_risk_assessment(stats: dict):
        """Render risk assessment."""
        st.subheader("Analisis Risiko")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Statistik Prediksi:**
            
            - Standard Deviasi: {stats['std']:,.2f}
            - Coefficient of Variation: {stats['cv']:.2f}%
            - Range Prediksi: {stats['range']:,.2f}
            - Expected Return: {stats['expected_return']:.2f}%
            """)
        
        with col2:
            # Risk level logic
            expected_return = abs(stats['expected_return'])
            
            if expected_return < 5:
                risk_level = "RENDAH"
                risk_color = "#22c55e" # Green
            elif expected_return < 10:
                risk_level = "SEDANG"
                risk_color = "#eab308" # Yellow
            else:
                risk_level = "TINGGI"
                risk_color = "#ef4444" # Red
            
            # Menggunakan HTML container untuk menggantikan emoji
            st.markdown(f"""
            <div style="padding: 1rem; border: 1px solid rgba(250,250,250,0.2); border-radius: 0.5rem; background-color: rgba(100,100,100,0.1);">
                <h4 style="margin-top:0">Penilaian Risiko</h4>
                <p>Level Risiko: <span style="color: {risk_color}; font-weight: bold; padding: 2px 6px; border: 1px solid {risk_color}; border-radius: 4px;">{risk_level}</span></p>
                <p>Volatilitas Model: {stats['cv']:.2f}%</p>
                <p>Confidence Level: 95%</p>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_disclaimer():
        """Render disclaimer."""
        st.warning("""
        **DISCLAIMER:**
        
        - Prediksi ini hanya untuk tujuan edukasi dan penelitian
        - Tidak boleh digunakan sebagai saran investasi
        - Pasar saham sangat volatil dan dipengaruhi banyak faktor eksternal
        - Selalu lakukan riset mendalam sebelum mengambil keputusan investasi
        - Konsultasikan dengan penasihat keuangan profesional
        """)