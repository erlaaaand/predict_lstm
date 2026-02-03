# User Guide - Stock Prediction Application

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Understanding Results](#understanding-results)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
# or
python run.py
```

4. Open your browser to `http://localhost:8501`

## Interface Overview

### Sidebar

The sidebar contains all configuration parameters:

**Data Parameters:**
- **Ticker Saham**: Stock symbol (e.g., BBCA.JK, TLKM.JK, AAPL)
- **Jumlah Data**: Number of trading days to fetch (90-730 days)
- **Jumlah Hari Prediksi**: Number of days to predict (1-60 days)

**Model Parameters:**
- **Tipe Model**: LSTM architecture type
  - Standard LSTM: Basic 2-layer LSTM
  - Bidirectional LSTM: Processes sequences both ways (recommended)
  - Deep LSTM: 3-layer deep architecture
- **Lookback Period**: Historical days used for prediction (10-90)

**Advanced Settings** (expandable):
- LSTM Units (Layer 1 & 2)
- Dropout Rate
- Epochs
- Batch Size
- Learning Rate

### Main Area

Five tabs display different aspects of the analysis:

1. **📊 Data & Statistik**: Raw data and basic statistics
2. **📈 Analisis Teknikal**: Charts with technical indicators
3. **🤖 Model LSTM**: Training results and performance
4. **🔮 Prediksi**: Future price predictions
5. **💾 Export**: Download data and reports

## Step-by-Step Tutorial

### Basic Analysis

1. **Select a Stock**
   - Enter ticker symbol in sidebar (e.g., `BBCA.JK` for Bank Central Asia)
   - Choose data period (recommended: 365 days)

2. **Configure Model** (use defaults for first try)
   - Model Type: Bidirectional LSTM
   - Lookback: 60 days
   - Keep other settings as default

3. **Start Analysis**
   - Click "🚀 Mulai Analisis"
   - Wait for data fetching and model training (2-5 minutes)

4. **Explore Results**
   - Navigate through tabs to see different visualizations
   - Check model performance metrics
   - View predictions

### Advanced Configuration

For better results, consider:

1. **More Data** = Better Training
   - Use 365-730 days for stable models
   - Minimum 180 days recommended

2. **Adjust Lookback**
   - Short-term predictions: 30-45 days
   - Long-term predictions: 60-90 days

3. **Model Architecture**
   - Volatile stocks: Use Deep LSTM
   - Stable stocks: Bidirectional LSTM works well

4. **Fine-tuning**
   - Increase epochs if underfitting (80-150)
   - Increase dropout if overfitting (0.2-0.4)

## Understanding Results

### Data & Statistik Tab

- **Recent Data**: Last 15 trading days
- **Descriptive Stats**: Min, max, mean, median, std dev
- **Latest Metrics**: Current price with change percentage

### Analisis Teknikal Tab

**Candlestick Chart:**
- Green candles: Price increased
- Red candles: Price decreased
- MA7, MA21, MA50: Moving averages

**Volume:**
- High volume with price increase: Strong buying
- High volume with price decrease: Strong selling

**RSI (Relative Strength Index):**
- Above 70: Overbought (might decrease)
- Below 30: Oversold (might increase)
- 30-70: Normal range

### Model LSTM Tab

**Training Curves:**
- Loss should decrease and converge
- Validation loss should follow training loss
- Large gap = overfitting

**Metrics:**
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)

**Good Model Indicators:**
- Test MAPE < 5%: Excellent
- Test MAPE 5-10%: Good
- Test MAPE 10-15%: Fair
- Test MAPE > 15%: Poor
- R² > 0.85: Excellent fit

### Prediksi Tab

**Statistics:**
- Final predicted price
- Average, min, max predictions
- Expected return percentage

**Chart:**
- Blue line: Historical data
- Red dashed line: Predictions
- Shaded area: 95% confidence interval

**Risk Assessment:**
- 🟢 Low Risk: < 5% expected return
- 🟡 Medium Risk: 5-10% expected return
- 🔴 High Risk: > 10% expected return

## Tips & Best Practices

### Data Selection

✅ **Do:**
- Use at least 365 days of data
- Check data quality before training
- Use liquid stocks with regular trading

❌ **Don't:**
- Use stocks with sparse trading
- Rely on less than 90 days of data
- Ignore data validation warnings

### Model Configuration

✅ **Do:**
- Start with default parameters
- Adjust gradually based on results
- Monitor validation loss during training

❌ **Don't:**
- Over-complicate with too many layers
- Set dropout too high (> 0.5)
- Train for too many epochs without early stopping

### Prediction Interpretation

✅ **Do:**
- Use predictions as one of many factors
- Consider confidence intervals
- Monitor model performance metrics
- Combine with fundamental analysis

❌ **Don't:**
- Rely solely on predictions for investing
- Ignore disclaimer warnings
- Trade based on predictions alone
- Assume 100% accuracy

## Troubleshooting

### Common Issues

**"Tidak ada data untuk ticker"**
- Solution: Check ticker symbol spelling
- Ensure ticker includes exchange suffix (.JK for Indonesia)

**"Data tidak cukup"**
- Solution: Increase data period
- Check if stock has enough trading history

**"Model loss not decreasing"**
- Solution: Adjust learning rate
- Increase model complexity
- Check data quality

**Predictions seem unrealistic**
- Solution: Retrain with more data
- Adjust lookback period
- Check for data anomalies

**Application runs slowly**
- Solution: Reduce data period
- Lower number of epochs
- Close other applications

### Getting Help

If you encounter issues:
1. Check error messages carefully
2. Review this guide
3. Verify installation of dependencies
4. Check data quality and parameters
5. Try with default settings first

## Disclaimer

⚠️ **Important Notice:**

This application is for **educational and research purposes only**.

- Predictions are not financial advice
- Past performance does not guarantee future results
- Stock markets are inherently volatile
- Always do your own research
- Consult with licensed financial advisors
- Never invest more than you can afford to lose

The creators of this application are not responsible for any financial losses incurred from using these predictions.
