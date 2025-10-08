import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import io
import warnings
import time
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Saham LSTM - Advanced Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 2rem;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# Header
st.markdown('<div class="main-header">Aplikasi Prediksi Harga Saham dengan LSTM Neural Network</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 PENGATURAN ANALISIS")
    
    st.subheader("📊 Parameter Data")
    ticker_input = st.text_input(
        "Ticker Saham",
        value="BBCA.JK",
        help="Masukkan kode ticker saham (contoh: BBCA.JK, TLKM.JK)"
    ).upper()
    
    n_days_data = st.slider(
        "Jumlah Data Trading (hari perdagangan)",
        min_value=90,
        max_value=730,
        value=365,
        step=30,
        help="Jumlah hari perdagangan yang akan diambil (tidak termasuk hari libur)"
    )
    
    n_days_predict = st.slider(
        "Jumlah Hari Prediksi",
        min_value=1,
        max_value=60,
        value=14,
        step=1,
        help="Prediksi untuk beberapa hari perdagangan ke depan"
    )
    
    st.markdown("---")
    st.subheader("🤖 Parameter Model LSTM")
    
    model_type = st.selectbox(
        "Tipe Model",
        ["Standard LSTM", "Bidirectional LSTM", "Deep LSTM"],
        help="Bidirectional LSTM biasanya lebih akurat untuk data saham"
    )
    
    lookback = st.slider(
        "Lookback Period",
        min_value=10,
        max_value=90,
        value=60,
        step=5,
        help="Jumlah hari historis yang digunakan untuk prediksi"
    )
    
    lstm_units = st.slider(
        "LSTM Units (Layer 1)",
        min_value=64,
        max_value=256,
        value=128,
        step=32
    )
    
    lstm_units_2 = st.slider(
        "LSTM Units (Layer 2)",
        min_value=32,
        max_value=128,
        value=64,
        step=16
    )
    
    dropout_rate = st.slider(
        "Dropout Rate",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.1,
        help="Mencegah overfitting"
    )
    
    epochs = st.slider(
        "Epochs",
        min_value=50,
        max_value=200,
        value=100,
        step=25
    )
    
    batch_size = st.selectbox(
        "Batch Size",
        [16, 32, 64, 128],
        index=1
    )
    
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001
    )
    
    st.markdown("---")
    analyze_button = st.button("🚀 MULAI ANALISIS", type="primary", use_container_width=True)
    
    # Reset button
    if st.button("🔄 Reset Model", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# Fungsi utility
@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, n_trading_days):
    """
    Mengambil data saham dengan jumlah hari perdagangan yang tepat
    """
    try:
        # Estimasi periode untuk mendapatkan n_trading_days
        # Asumsi: rata-rata 252 hari trading dalam setahun (365 hari)
        factor = 1.5  # Faktor pengaman untuk memastikan data cukup
        estimated_calendar_days = int(n_trading_days * 365/252 * factor)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=estimated_calendar_days)
        
        # Download data
        with st.spinner(f"Mengunduh data {ticker}..."):
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True,  # Auto adjust untuk split dan dividen
                progress=False,
                threads=False  # Disable threading untuk stabilitas
            )
            
            data.index = pd.to_datetime(data.index)
        
        if data.empty:
            st.error(f"Tidak ada data untuk ticker {ticker}")
            return None
        
        # Handle multi-level columns jika ada
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Clean data
        data = data.dropna()
        
        # Pastikan kita punya cukup data
        if len(data) < n_trading_days:
            # Jika data kurang, ambil periode lebih panjang
            extended_days = estimated_calendar_days * 2
            start_date = end_date - timedelta(days=extended_days)
            
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                auto_adjust=True,
                progress=False,
                threads=False
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.dropna()
        
        # Ambil tepat n_trading_days terakhir
        data = data.tail(n_trading_days)
        
        # Validasi data
        if len(data) < n_trading_days:
            st.warning(f"⚠️ Hanya tersedia {len(data)} hari perdagangan untuk {ticker}")
        
        # Tambahkan technical indicators dengan error handling
        try:
            # Simple Moving Averages
            data['MA7'] = data['Close'].rolling(window=7, min_periods=1).mean()
            data['MA21'] = data['Close'].rolling(window=21, min_periods=1).mean()
            data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            
            # Returns dan Volatility
            data['Returns'] = data['Close'].pct_change().fillna(0)
            data['Volatility'] = data['Returns'].rolling(window=20, min_periods=1).std()
            
            # RSI (Relative Strength Index)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Volume indicators
            data['Volume_MA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA'].replace(0, 1)
            
        except Exception as e:
            st.warning(f"⚠️ Beberapa indikator teknikal tidak dapat dihitung: {str(e)}")
        
        # Fill NaN values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
        
    except Exception as e:
        st.error(f"❌ Error mengambil data: {str(e)}")
        st.info("💡 Tips: Periksa koneksi internet dan validitas ticker saham")
        return None

def validate_data_for_training(data, lookback):
    """
    Validasi data sebelum training
    """
    if data is None:
        return False, "Data tidak tersedia"
    
    if len(data) < lookback + 20:  # Minimal data untuk train/test split
        return False, f"Data tidak cukup. Minimal {lookback + 20} data diperlukan, hanya tersedia {len(data)}"
    
    if data['Close'].isna().any():
        return False, "Terdapat nilai NaN dalam data Close price"
    
    if (data['Close'] <= 0).any():
        return False, "Terdapat nilai negatif atau nol dalam data Close price"
    
    return True, "Data valid"

def create_sequences(data, lookback):
    """
    Create sequences for LSTM with error handling
    """
    try:
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    except Exception as e:
        st.error(f"Error creating sequences: {str(e)}")
        return None, None

def build_lstm_model(lookback, units_1, units_2, dropout, model_type, lr):
    """
    Build LSTM model with error handling
    """
    try:
        model = Sequential()
        
        if model_type == "Bidirectional LSTM":
            model.add(Bidirectional(LSTM(units=units_1, return_sequences=True), 
                                   input_shape=(lookback, 1)))
            model.add(Dropout(dropout))
            model.add(Bidirectional(LSTM(units=units_2, return_sequences=False)))
            model.add(Dropout(dropout))
            
        elif model_type == "Deep LSTM":
            model.add(LSTM(units=units_1, return_sequences=True, 
                          input_shape=(lookback, 1)))
            model.add(Dropout(dropout))
            model.add(LSTM(units=units_2, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(LSTM(units=max(32, units_2//2), return_sequences=False))
            model.add(Dropout(dropout))
            
        else:  # Standard LSTM
            model.add(LSTM(units=units_1, return_sequences=True, 
                          input_shape=(lookback, 1)))
            model.add(Dropout(dropout))
            model.add(LSTM(units=units_2, return_sequences=False))
            model.add(Dropout(dropout))
        
        model.add(Dense(units=50, activation='relu'))
        model.add(Dropout(dropout/2))
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1))
        
        optimizer = Adam(learning_rate=lr, clipnorm=1.0)  # Add gradient clipping
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])  # Huber loss lebih robust
        
        return model
        
    except Exception as e:
        st.error(f"Error building model: {str(e)}")
        return None

def predict_future(model, last_sequence, scaler, n_days, lookback):
    """
    Predict future prices with error handling
    """
    try:
        print(f"DEBUG: Starting prediction for {n_days} days")
        predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(n_days):
            print(f"DEBUG: Predicting day {i+1}")
            pred = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
            predictions.append(pred[0, 0])
            current_sequence = np.append(current_sequence[1:], pred[0, 0])
        
        print("DEBUG: Inverse transforming predictions")
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        print("DEBUG: Prediction complete")
        return predictions.flatten()
        
    except Exception as e:
        print(f"DEBUG ERROR in predict_future: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"Error in prediction: {str(e)}")
        return None

def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics with error handling
    """
    try:
        # Remove any NaN or inf values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return 0, 0, 0, 0
        
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # MAPE with zero handling
        mape_mask = y_true_clean != 0
        if mape_mask.any():
            mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / 
                                 y_true_clean[mape_mask])) * 100
        else:
            mape = 0
        
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return rmse, mae, mape, r2
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return 0, 0, 0, 0

# Main Logic
if analyze_button:
    # Reset previous results
    st.session_state.model_trained = False
    
    # Validasi input
    if not ticker_input:
        st.error("❌ Silakan masukkan ticker saham")
        st.stop()
    
    # Get data
    with st.spinner(f"📊 Mengambil {n_days_data} hari perdagangan untuk {ticker_input}..."):
        stock_data = get_stock_data(ticker_input, n_days_data)
    
    # Validate data
    is_valid, message = validate_data_for_training(stock_data, lookback)
    
    if not is_valid:
        st.error(f"❌ {message}")
        st.stop()
    
    st.success(f"✅ Berhasil mengambil {len(stock_data)} hari perdagangan dari {stock_data.index[0].date()} hingga {stock_data.index[-1].date()}")
    
    # Store in session state
    st.session_state.stock_data = stock_data
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data & Statistik",
        "📈 Analisis Teknikal",
        "🤖 Model LSTM",
        "🔮 Prediksi",
        "💾 Export Data"
    ])
    
    with tab1:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader("Data Harga Saham Terbaru")
            try:
                display_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(15)
                st.dataframe(display_data.style.format({
                    'Open': '{:,.2f}',
                    'High': '{:,.2f}',
                    'Low': '{:,.2f}',
                    'Close': '{:,.2f}',
                    'Volume': '{:,.0f}'
                }), use_container_width=True, height=400)
            except Exception as e:
                st.error(f"Error displaying data: {str(e)}")
        
        with col2:
            st.subheader("Statistik Deskriptif")
            try:
                stats_df = pd.DataFrame({
                    'Metrik': ['Min', 'Max', 'Mean', 'Median', 'Std Dev', 'Data Count'],
                    'Nilai': [
                        f"{stock_data['Close'].min():,.2f}",
                        f"{stock_data['Close'].max():,.2f}",
                        f"{stock_data['Close'].mean():,.2f}",
                        f"{stock_data['Close'].median():,.2f}",
                        f"{stock_data['Close'].std():,.2f}",
                        f"{len(stock_data)}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
        
        with col3:
            st.subheader("Harga Terakhir")
            try:
                last_close = stock_data['Close'].iloc[-1]
                prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else last_close
                change = last_close - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                
                st.metric(
                    "Close Price",
                    f"{last_close:,.2f}",
                    f"{change:,.2f} ({change_pct:.2f}%)"
                )
                
                if 'Volume' in stock_data.columns:
                    vol_current = stock_data['Volume'].iloc[-1]
                    vol_prev = stock_data['Volume'].iloc[-2] if len(stock_data) > 1 else vol_current
                    vol_change = ((vol_current / vol_prev - 1) * 100) if vol_prev != 0 else 0
                    
                    st.metric(
                        "Volume",
                        f"{vol_current:,.0f}",
                        f"{vol_change:.2f}%"
                    )
                
                # Volatilitas
                if 'Returns' in stock_data.columns:
                    volatility = stock_data['Returns'].std() * np.sqrt(252) * 100
                    st.metric("Volatilitas Tahunan", f"{volatility:.2f}%")
                    
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
    
    with tab2:
        st.subheader("Grafik Harga dan Indikator Teknikal")
        
        try:
            # Candlestick dengan Moving Averages
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Harga Saham dengan Moving Averages', 'Volume Perdagangan', 'RSI'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Moving Averages
            if 'MA7' in stock_data.columns:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['MA7'],
                              name='MA7', line=dict(color='orange', width=1)),
                    row=1, col=1
                )
            if 'MA21' in stock_data.columns:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['MA21'],
                              name='MA21', line=dict(color='blue', width=1)),
                    row=1, col=1
                )
            if 'MA50' in stock_data.columns:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['MA50'],
                              name='MA50', line=dict(color='red', width=1)),
                    row=1, col=1
                )
            
            # Volume
            colors = ['red' if row['Close'] < row['Open'] else 'green' 
                     for idx, row in stock_data.iterrows()]
            fig.add_trace(
                go.Bar(x=stock_data.index, y=stock_data['Volume'],
                      name='Volume', marker_color=colors, showlegend=False),
                row=2, col=1
            )
            
            # RSI
            if 'RSI' in stock_data.columns:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['RSI'],
                              name='RSI', line=dict(color='purple')),
                    row=3, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Returns Distribution
            col1, col2 = st.columns(2)
            with col1:
                if 'Returns' in stock_data.columns:
                    fig_returns = go.Figure()
                    returns_data = stock_data['Returns'].dropna() * 100
                    fig_returns.add_trace(go.Histogram(
                        x=returns_data,
                        nbinsx=50,
                        name='Returns',
                        marker_color='steelblue'
                    ))
                    fig_returns.update_layout(
                        title='Distribusi Returns Harian (%)',
                        xaxis_title='Returns (%)',
                        yaxis_title='Frekuensi',
                        height=400
                    )
                    st.plotly_chart(fig_returns, use_container_width=True)
            
            with col2:
                if 'Volatility' in stock_data.columns:
                    fig_vol = go.Figure()
                    vol_data = stock_data['Volatility'].dropna() * 100
                    fig_vol.add_trace(go.Scatter(
                        x=stock_data.index[-len(vol_data):],
                        y=vol_data,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='coral'),
                        name='Volatilitas'
                    ))
                    fig_vol.update_layout(
                        title='Volatilitas Rolling (20 hari)',
                        xaxis_title='Tanggal',
                        yaxis_title='Volatilitas (%)',
                        height=400
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error creating charts: {str(e)}")
    
    with tab3:
        st.subheader("🤖 Pelatihan Model LSTM")
        
        try:
            with st.spinner("Mempersiapkan data untuk training..."):
                # Persiapan data
                close_prices = stock_data['Close'].values.reshape(-1, 1)
                
                # Pastikan stock_data.index adalah DatetimeIndex yang benar
                print(f"DEBUG: stock_data.index type: {type(stock_data.index)}")
                print(f"DEBUG: stock_data.index[-1] type: {type(stock_data.index[-1])}")
                
                # Scaling
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_prices)
                
                # Create sequences
                X, y = create_sequences(scaled_data, lookback)
                
                if X is None or y is None:
                    st.error("Failed to create sequences")
                    st.stop()
                
                # Validate sequences
                if len(X) < 50:
                    st.error(f"Data tidak cukup untuk training. Hanya {len(X)} sequences tersedia.")
                    st.stop()
                
                # Split data
                train_size = int(0.8 * len(X))
                val_size = int(0.1 * len(X))
                
                X_train = X[:train_size]
                y_train = y[:train_size]
                X_val = X[train_size:train_size+val_size]
                y_val = y[train_size:train_size+val_size]
                X_test = X[train_size+val_size:]
                y_test = y[train_size+val_size:]
                
                st.info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Build model
            model = build_lstm_model(
                lookback, lstm_units, lstm_units_2,
                dropout_rate, model_type, learning_rate
            )
            
            if model is None:
                st.error("Failed to build model")
                st.stop()
            
            # Training
            with st.spinner("🎯 Melatih model LSTM... Proses ini dapat memakan waktu beberapa menit..."):
                # Callbacks
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=0
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.00001,
                    verbose=0
                )
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom callback for progress
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = min(int((epoch + 1) / epochs * 100), 100)
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Epoch {epoch+1}/{epochs} - "
                            f"Loss: {logs['loss']:.6f} - "
                            f"Val Loss: {logs['val_loss']:.6f}"
                        )
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, reduce_lr, ProgressCallback()],
                    verbose=0
                )
                
                progress_bar.progress(100)
                status_text.success("✅ Pelatihan selesai!")
                
                # Store in session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.history = history
                st.session_state.model_trained = True
            
            # Model evaluation
            st.subheader("📊 Evaluasi Model")
            
            # Training curves
            # Training curves
            col1, col2 = st.columns(2)
            
            with col1:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.history['loss'],
                    name='Training Loss',
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.history['val_loss'],
                    name='Validation Loss',
                    mode='lines',
                    line=dict(color='red')
                ))
                fig_loss.update_layout(
                    title='Model Loss During Training',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=400
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                fig_mae = go.Figure()
                if 'mae' in history.history:
                    fig_mae.add_trace(go.Scatter(
                        y=history.history['mae'],
                        name='Training MAE',
                        mode='lines',
                        line=dict(color='green')
                    ))
                if 'val_mae' in history.history:
                    fig_mae.add_trace(go.Scatter(
                        y=history.history['val_mae'],
                        name='Validation MAE',
                        mode='lines',
                        line=dict(color='orange')
                    ))
                fig_mae.update_layout(
                    title='Mean Absolute Error During Training',
                    xaxis_title='Epoch',
                    yaxis_title='MAE',
                    height=400
                )
                st.plotly_chart(fig_mae, use_container_width=True)
            
            # Predictions on test set
            st.subheader("🎯 Performa pada Test Set")
            
            try:
                # Make predictions
                train_predictions = model.predict(X_train, verbose=0)
                val_predictions = model.predict(X_val, verbose=0)
                test_predictions = model.predict(X_test, verbose=0)
                
                # Inverse transform
                train_predictions = scaler.inverse_transform(train_predictions)
                val_predictions = scaler.inverse_transform(val_predictions)
                test_predictions = scaler.inverse_transform(test_predictions)
                
                y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
                # Calculate metrics
                train_rmse, train_mae, train_mape, train_r2 = calculate_metrics(
                    y_train_actual.flatten(), train_predictions.flatten()
                )
                val_rmse, val_mae, val_mape, val_r2 = calculate_metrics(
                    y_val_actual.flatten(), val_predictions.flatten()
                )
                test_rmse, test_mae, test_mape, test_r2 = calculate_metrics(
                    y_test_actual.flatten(), test_predictions.flatten()
                )
                
                # Store metrics
                st.session_state.metrics = {
                    'train': {'rmse': train_rmse, 'mae': train_mae, 'mape': train_mape, 'r2': train_r2},
                    'val': {'rmse': val_rmse, 'mae': val_mae, 'mape': val_mape, 'r2': val_r2},
                    'test': {'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape, 'r2': test_r2}
                }
                
                # Display metrics
                metrics_df = pd.DataFrame({
                    'Dataset': ['Training', 'Validation', 'Test'],
                    'RMSE': [train_rmse, val_rmse, test_rmse],
                    'MAE': [train_mae, val_mae, test_mae],
                    'MAPE (%)': [train_mape, val_mape, test_mape],
                    'R²': [train_r2, val_r2, test_r2]
                })
                
                st.dataframe(
                    metrics_df.style.format({
                        'RMSE': '{:,.2f}',
                        'MAE': '{:,.2f}',
                        'MAPE (%)': '{:.2f}',
                        'R²': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                # Visualize predictions vs actual
                st.subheader("📈 Prediksi vs Aktual")
                
                # Prepare data for visualization
                train_dates = stock_data.index[lookback:train_size+lookback]
                val_dates = stock_data.index[train_size+lookback:train_size+val_size+lookback]
                test_dates = stock_data.index[train_size+val_size+lookback:train_size+val_size+len(X_test)+lookback]
                
                fig_pred = go.Figure()
                
                # Actual prices
                fig_pred.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name='Actual',
                    mode='lines',
                    line=dict(color='black', width=1)
                ))
                
                # Training predictions
                fig_pred.add_trace(go.Scatter(
                    x=train_dates,
                    y=train_predictions.flatten(),
                    name='Train Predictions',
                    mode='lines',
                    line=dict(color='blue', width=2)
                ))
                
                # Validation predictions
                fig_pred.add_trace(go.Scatter(
                    x=val_dates,
                    y=val_predictions.flatten(),
                    name='Val Predictions',
                    mode='lines',
                    line=dict(color='orange', width=2)
                ))
                
                # Test predictions
                fig_pred.add_trace(go.Scatter(
                    x=test_dates,
                    y=test_predictions.flatten(),
                    name='Test Predictions',
                    mode='lines',
                    line=dict(color='red', width=2)
                ))
                
                fig_pred.update_layout(
                    title='Prediksi Model vs Harga Aktual',
                    xaxis_title='Tanggal',
                    yaxis_title='Harga (IDR)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Residual analysis
                st.subheader("📊 Analisis Residual")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Residual plot
                    test_residuals = y_test_actual.flatten() - test_predictions.flatten()
                    
                    fig_residual = go.Figure()
                    fig_residual.add_trace(go.Scatter(
                        x=test_predictions.flatten(),
                        y=test_residuals,
                        mode='markers',
                        marker=dict(color='purple', size=5),
                        name='Residuals'
                    ))
                    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_residual.update_layout(
                        title='Residual Plot (Test Set)',
                        xaxis_title='Predicted Values',
                        yaxis_title='Residuals',
                        height=400
                    )
                    st.plotly_chart(fig_residual, use_container_width=True)
                
                with col2:
                    # Residual histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=test_residuals,
                        nbinsx=30,
                        name='Residuals',
                        marker_color='teal'
                    ))
                    fig_hist.update_layout(
                        title='Distribusi Residual',
                        xaxis_title='Residual',
                        yaxis_title='Frekuensi',
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Success message
                st.success(f"""
                ✅ **Model berhasil dilatih!**
                - Model Type: {model_type}
                - Total Parameters: {model.count_params():,}
                - Best Test MAPE: {test_mape:.2f}%
                - Best Test R²: {test_r2:.4f}
                """)
                
            except Exception as e:
                st.error(f"Error in model evaluation: {str(e)}")
                
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
    
    with tab4:
        st.subheader("🔮 Prediksi Harga Masa Depan")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Silakan latih model terlebih dahulu di tab 'Model LSTM'")
        else:
            try:
                with st.spinner(f"Membuat prediksi untuk {n_days_predict} hari ke depan..."):
                    st.write("DEBUG: Step 1 - Getting last sequence")
                    # Get last sequence
                    close_prices = st.session_state.stock_data['Close'].values.reshape(-1, 1)
                    scaler = st.session_state.scaler
                    scaled_data = scaler.transform(close_prices)
                    last_sequence = scaled_data[-lookback:]
                    
                    st.write("DEBUG: Step 2 - Calling predict_future")
                    # Make predictions
                    future_predictions = predict_future(
                        st.session_state.model,
                        last_sequence,
                        st.session_state.scaler,
                        n_days_predict,
                        lookback
                    )
                    
                    if future_predictions is None:
                        st.error("Failed to generate predictions")
                        st.stop()
                    
                    # --- PERBAIKAN DIMULAI DI SINI ---
                    # Generate future dates (trading days only)
                    last_date = st.session_state.stock_data.index[-1]
                    # Menggunakan pd.date_range dengan frekuensi 'B' (business day)
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days_predict, freq='B')
                    # --- PERBAIKAN SELESAI ---

                    # Konversi ke Python datetime (bukan pandas Timestamp)
                    if hasattr(last_date, 'to_pydatetime'):
                        last_date = last_date.to_pydatetime()
                    else:
                        last_date = datetime.fromisoformat(str(last_date))

                    future_dates = []
                    current_date = last_date

                    while len(future_dates) < n_days_predict:
                        current_date = current_date + timedelta(days=1)
                        # Skip weekend (Sabtu=5, Minggu=6)
                        if current_date.weekday() < 5:  # Senin-Jumat (0-4)
                            future_dates.append(current_date)

                    future_dates = pd.DatetimeIndex(future_dates)
                    
                    # Create prediction dataframe
                    prediction_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Price': future_predictions
                    })
                    prediction_df.set_index('Date', inplace=True)
                    
                    # Calculate prediction statistics
                    pred_mean = future_predictions.mean()
                    pred_std = future_predictions.std()
                    pred_min = future_predictions.min()
                    pred_max = future_predictions.max()
                    last_actual = stock_data['Close'].iloc[-1]
                    expected_return = ((future_predictions[-1] - last_actual) / last_actual) * 100
                    
                    # Display prediction statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Harga Akhir Prediksi",
                            f"{future_predictions[-1]:,.2f}",
                            f"{expected_return:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Rata-rata Prediksi",
                            f"{pred_mean:,.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Min Prediksi",
                            f"{pred_min:,.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Max Prediksi",
                            f"{pred_max:,.2f}"
                        )
                    
                    # Visualization
                    st.subheader("📈 Visualisasi Prediksi")
                    
                    # Combined chart
                    fig_future = go.Figure()
                    
                    # Historical data (last 60 days)
                    historical_days = min(60, len(stock_data))
                    fig_future.add_trace(go.Scatter(
                        x=stock_data.index[-historical_days:],
                        y=stock_data['Close'].iloc[-historical_days:],
                        name='Historical',
                        mode='lines',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predictions
                    fig_future.add_trace(go.Scatter(
                        x=prediction_df.index,
                        y=prediction_df['Predicted_Price'],
                        name='Predictions',
                        mode='lines+markers',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Add confidence interval (simplified - using std dev)
                    upper_bound = future_predictions + 2 * pred_std
                    lower_bound = future_predictions - 2 * pred_std
                    
                    fig_future.add_trace(go.Scatter(
                        x=prediction_df.index,
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig_future.add_trace(go.Scatter(
                        x=prediction_df.index,
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(255,0,0,0)',
                        name='95% Confidence',
                        fillcolor='rgba(255,0,0,0.1)'
                    ))
                    
                    # Add vertical line at prediction start
                    fig_future.add_vline(
                        x=stock_data.index[-1],
                        line_dash="dot",
                        line_color="gray",
                        annotation_text="Prediction Start"
                    )
                    
                    fig_future.update_layout(
                        title=f'Prediksi Harga {ticker_input} untuk {n_days_predict} Hari Kedepan',
                        xaxis_title='Tanggal',
                        yaxis_title='Harga (IDR)',
                        height=600,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_future, use_container_width=True)
                    
                    # Prediction table
                    st.subheader("📊 Tabel Prediksi Harian")
                    
                    # Add daily changes
                    prediction_display = prediction_df.copy()
                    prediction_display['Daily_Change'] = prediction_display['Predicted_Price'].diff()
                    prediction_display['Daily_Change_%'] = prediction_display['Predicted_Price'].pct_change() * 100

                    # ✅ PERBAIKAN: Gunakan nama kolom, bukan indeks
                    first_predicted_price = prediction_display['Predicted_Price'].iloc[0]
                    prediction_display.loc[prediction_display.index[0], 'Daily_Change'] = \
                        first_predicted_price - last_actual
                    prediction_display.loc[prediction_display.index[0], 'Daily_Change_%'] = \
                        (first_predicted_price - last_actual) / last_actual * 100
                    
                    st.dataframe(
                        prediction_display.style.format({
                            'Predicted_Price': '{:,.2f}',
                            'Daily_Change': '{:+,.2f}',
                            'Daily_Change_%': '{:+.2f}%'
                        }).background_gradient(subset=['Predicted_Price'], cmap='RdYlGn'),
                        use_container_width=True
                    )
                    
                    # Risk Assessment
                    st.subheader("⚠️ Analisis Risiko")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **Statistik Prediksi:**
                        - Standard Deviasi: {pred_std:,.2f}
                        - Coefficient of Variation: {(pred_std/pred_mean)*100:.2f}%
                        - Range Prediksi: {pred_max - pred_min:,.2f}
                        - Expected Return: {expected_return:.2f}%
                        """)
                    
                    with col2:
                        # Risk level assessment
                        if abs(expected_return) < 5:
                            risk_level = "Rendah"
                            risk_color = "green"
                        elif abs(expected_return) < 10:
                            risk_level = "Sedang"
                            risk_color = "yellow"
                        else:
                            risk_level = "Tinggi"
                            risk_color = "red"
                        
                        st.warning(f"""
                        **Penilaian Risiko:**
                        - Level Risiko: **:{risk_color}[{risk_level}]**
                        - Volatilitas Model: {(pred_std/pred_mean)*100:.2f}%
                        - Confidence Level: 95%
                        - Model R² Score: {st.session_state.metrics['test']['r2']:.4f}
                        """)
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **DISCLAIMER:**
                    - Prediksi ini hanya untuk tujuan edukasi dan penelitian
                    - Tidak boleh digunakan sebagai saran investasi
                    - Pasar saham sangat volatil dan dipengaruhi banyak faktor eksternal
                    - Selalu lakukan riset mendalam sebelum mengambil keputusan investasi
                    - Konsultasikan dengan penasihat keuangan profesional
                    """)
                    
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    
    with tab5:
        st.subheader("💾 Export Data dan Model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Export Data Historis")
            if stock_data is not None:
                csv_hist = stock_data.to_csv()
                st.download_button(
                    label="Download Historical Data (CSV)",
                    data=csv_hist,
                    file_name=f"{ticker_input}_historical_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("🔮 Export Prediksi")
            if st.session_state.model_trained:
                try:
                    # Generate predictions again for export
                    last_sequence = scaled_data[-lookback:]
                    future_predictions = predict_future(
                        st.session_state.model,
                        last_sequence,
                        st.session_state.scaler,
                        n_days_predict,
                        lookback
                    )
                    
                    if future_predictions is not None:
                        # ✅ PERBAIKAN: Gunakan cara yang sama seperti Tab4
                        last_date = st.session_state.stock_data.index[-1]
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days_predict, freq='B')
                        
                        # Konversi ke Python datetime
                        if hasattr(last_date, 'to_pydatetime'):
                            last_date = last_date.to_pydatetime()
                        else:
                            last_date = datetime.fromisoformat(str(last_date))
                        
                        # Generate future dates (trading days only)
                        future_dates = []
                        current_date = last_date
                        
                        while len(future_dates) < n_days_predict:
                            current_date = current_date + timedelta(days=1)
                            # Skip weekend
                            if current_date.weekday() < 5:  # Senin-Jumat (0-4)
                                future_dates.append(current_date)
                        
                        future_dates = pd.DatetimeIndex(future_dates)
                        
                        export_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted_Price': future_predictions
                        })
                        
                        csv_pred = export_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions (CSV)",
                            data=csv_pred,
                            file_name=f"{ticker_input}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Error exporting predictions: {str(e)}")
        
        with col3:
            st.subheader("📈 Export Report")
            if st.session_state.model_trained:
                try:
                    # Generate comprehensive report
                    report = f"""
LAPORAN ANALISIS PREDIKSI SAHAM
================================
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {ticker_input}

DATA HISTORIS
-------------
Periode: {stock_data.index[0].date()} s/d {stock_data.index[-1].date()}
Jumlah Data: {len(stock_data)} hari perdagangan
Harga Terakhir: {stock_data['Close'].iloc[-1]:,.2f}
Min Harga: {stock_data['Close'].min():,.2f}
Max Harga: {stock_data['Close'].max():,.2f}
Rata-rata: {stock_data['Close'].mean():,.2f}

PARAMETER MODEL
---------------
Model Type: {model_type}
Lookback Period: {lookback} hari
LSTM Units (Layer 1): {lstm_units}
LSTM Units (Layer 2): {lstm_units_2}
Dropout Rate: {dropout_rate}
Learning Rate: {learning_rate}
Epochs: {epochs}
Batch Size: {batch_size}

PERFORMA MODEL
--------------
Test RMSE: {st.session_state.metrics['test']['rmse']:,.2f}
Test MAE: {st.session_state.metrics['test']['mae']:,.2f}
Test MAPE: {st.session_state.metrics['test']['mape']:.2f}%
Test R²: {st.session_state.metrics['test']['r2']:.4f}

PREDIKSI {n_days_predict} HARI
-------------------------------
Harga Prediksi Akhir: {future_predictions[-1]:,.2f}
Expected Return: {expected_return:.2f}%
Min Prediksi: {pred_min:,.2f}
Max Prediksi: {pred_max:,.2f}
Rata-rata Prediksi: {pred_mean:,.2f}

DISCLAIMER
----------
Prediksi ini hanya untuk tujuan edukasi dan penelitian.
Tidak boleh digunakan sebagai saran investasi.
"""
                    st.download_button(
                        label="Download Full Report (TXT)",
                        data=report,
                        file_name=f"{ticker_input}_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Model Architecture Summary
        if st.session_state.model_trained:
            with st.expander("🏗️ Model Architecture Summary"):
                model_summary = []
                st.session_state.model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))

else:
    # Landing page when no analysis is running
    st.info("""
    👈 **Mulai dengan mengatur parameter di sidebar:**
    1. Pilih ticker saham yang ingin dianalisis
    2. Tentukan jumlah data historis yang akan digunakan
    3. Atur parameter model LSTM sesuai kebutuhan
    4. Klik tombol "MULAI ANALISIS" untuk memulai
    
    **Fitur Aplikasi:**
    - 📊 Analisis data historis dan teknikal saham
    - 🤖 Pelatihan model LSTM dengan berbagai arsitektur
    - 🔮 Prediksi harga saham untuk beberapa hari ke depan
    - 📈 Visualisasi interaktif dengan Plotly
    - 💾 Export data dan laporan analisis
    
    **Tips:**
    - Gunakan minimal 365 hari data untuk hasil yang lebih baik
    - Bidirectional LSTM umumnya memberikan hasil lebih akurat
    - Perhatikan metrik MAPE dan R² untuk menilai akurasi model
    """)
    
    # Sample tickers
    st.subheader("🏢 Contoh Ticker Saham Indonesia:")
    ticker_examples = pd.DataFrame({
        'Sektor': ['Banking', 'Telekomunikasi', 'Consumer', 'Mining', 'Property'],
        'Ticker': ['BBCA.JK', 'TLKM.JK', 'ICBP.JK', 'ADRO.JK', 'BSDE.JK'],
        'Nama Perusahaan': ['Bank Central Asia', 'Telkom Indonesia', 'Indofood CBP', 'Adaro Energy', 'BSD']
    })
    st.dataframe(ticker_examples, use_container_width=True, hide_index=True)      