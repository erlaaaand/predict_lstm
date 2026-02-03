"""
LSTM model architecture builder.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from typing import Optional


class LSTMModelBuilder:
    """Builds different types of LSTM models."""
    
    @staticmethod
    def build(
        lookback: int,
        units_1: int,
        units_2: int,
        dropout: float,
        model_type: str,
        learning_rate: float
    ) -> Optional[Sequential]:
        """
        Build LSTM model based on specified type.
        
        Args:
            lookback: Input sequence length
            units_1: Units in first LSTM layer
            units_2: Units in second LSTM layer
            dropout: Dropout rate
            model_type: Type of model (Standard/Bidirectional/Deep)
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model or None if error
        """
        try:
            model = Sequential()
            
            if model_type == "Bidirectional LSTM":
                model = LSTMModelBuilder._build_bidirectional(
                    model, lookback, units_1, units_2, dropout
                )
            elif model_type == "Deep LSTM":
                model = LSTMModelBuilder._build_deep(
                    model, lookback, units_1, units_2, dropout
                )
            else:  # Standard LSTM
                model = LSTMModelBuilder._build_standard(
                    model, lookback, units_1, units_2, dropout
                )
            
            # Add dense layers
            model.add(Dense(units=50, activation='relu'))
            model.add(Dropout(dropout / 2))
            model.add(Dense(units=25, activation='relu'))
            model.add(Dense(units=1))
            
            # Compile model
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
            model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
            
            return model
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            return None
    
    @staticmethod
    def _build_standard(
        model: Sequential,
        lookback: int,
        units_1: int,
        units_2: int,
        dropout: float
    ) -> Sequential:
        """Build standard LSTM architecture."""
        model.add(LSTM(
            units=units_1,
            return_sequences=True,
            input_shape=(lookback, 1)
        ))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units_2, return_sequences=False))
        model.add(Dropout(dropout))
        return model
    
    @staticmethod
    def _build_bidirectional(
        model: Sequential,
        lookback: int,
        units_1: int,
        units_2: int,
        dropout: float
    ) -> Sequential:
        """Build bidirectional LSTM architecture."""
        model.add(Bidirectional(
            LSTM(units=units_1, return_sequences=True),
            input_shape=(lookback, 1)
        ))
        model.add(Dropout(dropout))
        model.add(Bidirectional(
            LSTM(units=units_2, return_sequences=False)
        ))
        model.add(Dropout(dropout))
        return model
    
    @staticmethod
    def _build_deep(
        model: Sequential,
        lookback: int,
        units_1: int,
        units_2: int,
        dropout: float
    ) -> Sequential:
        """Build deep LSTM architecture with 3 LSTM layers."""
        model.add(LSTM(
            units=units_1,
            return_sequences=True,
            input_shape=(lookback, 1)
        ))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units_2, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=max(32, units_2 // 2), return_sequences=False))
        model.add(Dropout(dropout))
        return model
