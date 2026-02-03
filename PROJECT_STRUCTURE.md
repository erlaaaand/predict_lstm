# Project Structure

```
stock-prediction/
│
├── app/                              # Main application package
│   ├── __init__.py                  # Package initialization
│   │
│   ├── config/                      # Configuration module
│   │   ├── __init__.py
│   │   └── settings.py              # App settings & constants
│   │
│   ├── core/                        # Core business logic
│   │   ├── __init__.py
│   │   │
│   │   ├── data/                    # Data processing
│   │   │   ├── __init__.py
│   │   │   ├── fetcher.py           # Data fetching (yfinance)
│   │   │   ├── validator.py         # Data validation
│   │   │   ├── indicators.py        # Technical indicators
│   │   │   └── preprocessor.py      # ML preprocessing
│   │   │
│   │   ├── models/                  # ML models
│   │   │   ├── __init__.py
│   │   │   ├── builder.py           # LSTM architecture builder
│   │   │   ├── trainer.py           # Model training
│   │   │   └── predictor.py         # Prediction logic
│   │   │
│   │   └── utils/                   # Utilities
│   │       ├── __init__.py
│   │       └── metrics.py           # Evaluation metrics
│   │
│   ├── services/                    # Service/orchestration layer
│   │   ├── __init__.py
│   │   └── stock_service.py         # Main service orchestrator
│   │
│   └── ui/                          # User interface
│       ├── __init__.py
│       ├── styling.py               # Theme & CSS
│       ├── sidebar.py               # Sidebar component
│       ├── charts.py                # Reusable charts
│       │
│       └── pages/                   # Page components
│           ├── __init__.py
│           ├── landing.py           # Welcome page
│           ├── data_stats.py        # Data statistics
│           ├── technical_analysis.py # Technical charts
│           ├── model_training.py    # Training results
│           ├── prediction.py        # Predictions
│           └── export.py            # Export/download
│
├── main.py                          # Application entry point
├── run.py                           # Run script
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
│
├── README.md                        # Project overview
├── ARCHITECTURE.md                  # Architecture docs
├── USER_GUIDE.md                    # User documentation
├── CHANGELOG.md                     # Version history
└── PROJECT_STRUCTURE.md             # This file
```

## Module Descriptions

### Configuration (`app/config/`)
Centralized configuration management using dataclasses for type safety.

### Core (`app/core/`)

#### Data Module
- **fetcher.py**: Retrieves stock data from Yahoo Finance with caching
- **validator.py**: Validates data quality and integrity
- **indicators.py**: Calculates technical indicators (MA, RSI, MACD, etc.)
- **preprocessor.py**: Prepares data for machine learning (scaling, sequences)

#### Models Module
- **builder.py**: Builds different LSTM architectures (Standard, Bidirectional, Deep)
- **trainer.py**: Handles model training with callbacks and progress tracking
- **predictor.py**: Generates future price predictions

#### Utils Module
- **metrics.py**: Calculates evaluation metrics (RMSE, MAE, MAPE, R²)

### Services (`app/services/`)
Orchestration layer that coordinates core modules and provides high-level API.

### UI (`app/ui/`)

#### Components
- **styling.py**: Shadcn-inspired theme and CSS
- **sidebar.py**: Parameter configuration sidebar
- **charts.py**: Reusable Plotly chart components

#### Pages
- **landing.py**: Welcome screen with instructions
- **data_stats.py**: Data statistics and recent prices
- **technical_analysis.py**: Technical analysis charts
- **model_training.py**: Training curves and metrics
- **prediction.py**: Future predictions and risk assessment
- **export.py**: Data export and reports

## Design Patterns

### Separation of Concerns
Each module has a single, well-defined responsibility.

### Service Layer Pattern
Services orchestrate core modules and provide clean API to UI.

### Component-Based UI
Reusable UI components and page modules.

### Dependency Injection
Configuration and dependencies injected rather than hardcoded.

### Caching Strategy
Expensive operations cached using Streamlit's caching decorators.

## File Naming Conventions

- **Modules**: lowercase with underscores (`stock_service.py`)
- **Classes**: PascalCase (`StockAnalysisService`)
- **Functions**: lowercase with underscores (`fetch_stock_data`)
- **Constants**: UPPERCASE (`MAX_DATA_DAYS`)

## Import Guidelines

```python
# Standard library
import os
from datetime import datetime

# Third-party
import streamlit as st
import pandas as pd

# Local application
from app.config import config
from app.core.data import StockDataFetcher
from app.services import StockAnalysisService
```

## Code Organization Principles

1. **Single Responsibility**: Each class/function does one thing well
2. **DRY (Don't Repeat Yourself)**: Extract common functionality
3. **Modularity**: High cohesion, loose coupling
4. **Testability**: Easy to test each component independently
5. **Readability**: Clear, self-documenting code with comments
