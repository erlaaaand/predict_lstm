# Stock Price Prediction with LSTM

Aplikasi prediksi harga saham menggunakan LSTM (Long Short-Term Memory) Neural Network dengan interface yang clean dan modern.

## Fitur Utama

- 📊 Analisis data historis saham dengan berbagai indikator teknikal
- 🤖 Model LSTM yang dapat dikonfigurasi (Standard, Bidirectional, Deep)
- 🔮 Prediksi harga saham untuk N hari ke depan
- 📈 Visualisasi interaktif dengan Plotly
- 💾 Export data dan laporan analisis
- ⚡ Caching untuk performa optimal

## Struktur Project

```
stock-prediction/
├── app/
│   ├── config/           # Konfigurasi aplikasi
│   ├── core/            # Business logic inti
│   │   ├── data/        # Data processing & validation
│   │   ├── models/      # Model LSTM & training
│   │   └── utils/       # Utility functions
│   ├── services/        # Service layer
│   └── ui/              # UI components & pages
├── assets/              # Static assets
├── tests/               # Unit tests
├── main.py             # Entry point aplikasi
└── requirements.txt    # Dependencies
```

## Instalasi

```bash
# Clone repository
git clone <repository-url>
cd stock-prediction

# Install dependencies
pip install -r requirements.txt

# Run aplikasi
streamlit run main.py
```

## Penggunaan

1. Masukkan ticker saham (contoh: BBCA.JK, TLKM.JK)
2. Atur parameter data dan model LSTM
3. Klik "MULAI ANALISIS"
4. Eksplorasi hasil di berbagai tab yang tersedia

## Teknologi

- **Framework**: Streamlit
- **ML Framework**: TensorFlow/Keras
- **Data**: yfinance, pandas, numpy
- **Visualization**: Plotly
- **Styling**: Custom CSS (Shadcn-inspired)

## Disclaimer

Aplikasi ini hanya untuk tujuan edukasi dan penelitian. 
Tidak boleh digunakan sebagai saran investasi.
