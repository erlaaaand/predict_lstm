# Architecture Documentation

## Project Structure

```
stock-prediction/
├── app/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Centralized configuration
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── fetcher.py       # Data fetching from Yahoo Finance
│   │   │   ├── validator.py     # Data validation
│   │   │   ├── indicators.py    # Technical indicators calculation
│   │   │   └── preprocessor.py  # Data preprocessing for ML
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── builder.py       # LSTM model architecture builder
│   │   │   ├── trainer.py       # Model training logic
│   │   │   └── predictor.py     # Prediction logic
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── metrics.py       # Evaluation metrics
│   ├── services/
│   │   ├── __init__.py
│   │   └── stock_service.py     # Orchestration service
│   └── ui/
│       ├── __init__.py
│       ├── styling.py           # UI theme and styling
│       ├── sidebar.py           # Sidebar component
│       ├── charts.py            # Reusable chart components
│       └── pages/
│           ├── __init__.py
│           ├── landing.py       # Landing page
│           ├── data_stats.py    # Data statistics page
│           ├── technical_analysis.py  # Technical analysis page
│           ├── model_training.py      # Model training page
│           ├── prediction.py          # Prediction page
│           └── export.py             # Export page
├── main.py                      # Application entry point
├── requirements.txt
├── README.md
├── .gitignore
└── ARCHITECTURE.md
```

## Design Principles

### 1. Separation of Concerns

Each module has a single, well-defined responsibility:

- **Config**: Application configuration and constants
- **Core**: Business logic (data processing, models, utilities)
- **Services**: Orchestration layer that coordinates core modules
- **UI**: Presentation layer (styling, components, pages)

### 2. Modularity

- Each component is self-contained and can be tested independently
- Clear interfaces between modules
- Loose coupling between layers

### 3. Reusability

- Common functionality is extracted into utility modules
- Chart components are reusable across different pages
- Configuration is centralized for easy modification

### 4. Scalability

- Service layer makes it easy to add new features
- Modular architecture allows for easy component replacement
- Clear separation allows for horizontal scaling

## Data Flow

```
User Input (Sidebar)
       ↓
   main.py (Controller)
       ↓
StockAnalysisService (Orchestration)
       ↓
   ├── StockDataFetcher → Data Validation → Technical Indicators
   ├── DataPreprocessor → Model Builder → Model Trainer
   └── Model Predictor
       ↓
   UI Pages (Presentation)
```

## Key Components

### Configuration Layer (`app/config/`)

- Centralizes all application settings
- Uses dataclass for type safety
- Makes configuration changes easy

### Core Layer (`app/core/`)

#### Data Module
- **Fetcher**: Retrieves stock data from Yahoo Finance with caching
- **Validator**: Ensures data quality and integrity
- **Indicators**: Calculates technical indicators (MA, RSI, etc.)
- **Preprocessor**: Prepares data for ML (scaling, sequences)

#### Models Module
- **Builder**: Creates different LSTM architectures
- **Trainer**: Handles model training with callbacks
- **Predictor**: Generates future price predictions

#### Utils Module
- **Metrics**: Calculates evaluation metrics (RMSE, MAE, MAPE, R²)

### Service Layer (`app/services/`)

- Orchestrates data fetching, model training, and prediction
- Provides high-level API for the UI layer
- Handles error management and logging

### UI Layer (`app/ui/`)

#### Styling
- Clean, modern design inspired by Shadcn
- Consistent color palette and typography
- Responsive layout

#### Components
- **Sidebar**: Parameter configuration
- **Charts**: Reusable Plotly visualizations

#### Pages
- Modular page components for each tab
- Clear separation of concerns
- Easy to add new pages

## Technology Stack

- **Framework**: Streamlit
- **ML**: TensorFlow/Keras
- **Data**: pandas, numpy, yfinance
- **Visualization**: Plotly
- **Preprocessing**: scikit-learn

## Best Practices

1. **Type Hints**: Use Python type hints for better code clarity
2. **Docstrings**: Document all public methods
3. **Error Handling**: Graceful error handling at all levels
4. **Caching**: Use Streamlit caching for expensive operations
5. **Session State**: Proper management of application state
6. **Clean Code**: Follow PEP 8 and maintain readability

## Future Enhancements

- [ ] Unit tests for all modules
- [ ] Model versioning and experiment tracking
- [ ] Multi-ticker analysis
- [ ] Real-time data streaming
- [ ] Advanced technical indicators
- [ ] Portfolio optimization
- [ ] Backtesting framework
- [ ] API endpoint for predictions
