"""
LSTM model architecture builder with regularization.
Fixed version supporting multi-feature input.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Optional, Tuple, List
import numpy as np
from config.settings import config


class RegularizedLSTMBuilder:
    """Build regularized LSTM models."""
    
    @staticmethod
    def build_bidirectional_model(
        input_shape: Tuple[int, int],
        units_1: int = 128,
        units_2: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ) -> Sequential:
        """Build bidirectional LSTM with regularization."""
        try:
            model = Sequential([
                # First Bidirectional LSTM layer
                Bidirectional(
                    LSTM(
                        units=units_1,
                        return_sequences=True,
                        recurrent_dropout=0.1,
                        kernel_regularizer=l1_l2(
                            l1=config.L1_REGULARIZATION,
                            l2=config.L2_REGULARIZATION
                        )
                    ),
                    input_shape=input_shape
                ),
                Dropout(dropout),
                BatchNormalization(),
                
                # Second Bidirectional LSTM layer
                Bidirectional(
                    LSTM(
                        units=units_2,
                        return_sequences=False,
                        recurrent_dropout=0.1,
                        kernel_regularizer=l1_l2(
                            l1=config.L1_REGULARIZATION,
                            l2=config.L2_REGULARIZATION
                        )
                    )
                ),
                Dropout(dropout),
                BatchNormalization(),
                
                # Dense layers
                Dense(
                    units=50,
                    activation='relu',
                    kernel_regularizer=l1_l2(
                        l1=config.L1_REGULARIZATION,
                        l2=config.L2_REGULARIZATION
                    )
                ),
                Dropout(dropout / 2),
                
                Dense(units=25, activation='relu'),
                Dropout(dropout / 2),
                
                # Output layer
                Dense(units=1)
            ])
            
            # Compile
            optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
            model.compile(
                optimizer=optimizer,
                loss='huber',
                metrics=['mae', 'mse']
            )
            
            return model
            
        except Exception as e:
            raise ValueError(f"Error building model: {str(e)}")
    
    @staticmethod
    def get_callbacks() -> List:
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LEARNING_RATE,
                verbose=1
            )
        ]
        return callbacks


class EnsembleModel:
    """Ensemble of LSTM models for improved predictions."""
    
    def __init__(self, n_models: int = 3):
        self.n_models = n_models
        self.models = []
    
    def build_ensemble(
        self,
        input_shape: Tuple[int, int],
        units_1: int = 128,
        units_2: int = 64,
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ):
        """Build ensemble of models with variation."""
        self.models = []
        
        for i in range(self.n_models):
            # Vary hyperparameters slightly
            units_1_var = int(units_1 * (0.9 + 0.2 * i / self.n_models))
            units_2_var = int(units_2 * (0.9 + 0.2 * i / self.n_models))
            dropout_var = dropout * (0.9 + 0.2 * i / self.n_models)
            
            model = RegularizedLSTMBuilder.build_bidirectional_model(
                input_shape=input_shape,
                units_1=units_1_var,
                units_2=units_2_var,
                dropout=dropout_var,
                learning_rate=learning_rate
            )
            
            self.models.append(model)
    
    def train_ensemble(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 0
    ):
        """Train all models in ensemble."""
        histories = []
        callbacks = RegularizedLSTMBuilder.get_callbacks()
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}...")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            histories.append(history)
        
        return histories
    
    def predict(self, X, verbose: int = 0):
        """Average predictions from all models."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X, verbose=verbose)
            predictions.append(pred)
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction
    
    def count_params(self):
        """Count total parameters."""
        if len(self.models) == 0:
            return 0
        return self.models[0].count_params() * self.n_models


class LSTMModelBuilder:
    """Legacy model builder for backward compatibility."""
    
    @staticmethod
    def build(
        lookback: int,
        units_1: int,
        units_2: int,
        dropout: float,
        model_type: str,
        learning_rate: float,
        n_features: int = 1
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
            n_features: Number of input features
            
        Returns:
            Compiled Keras model or None if error
        """
        try:
            input_shape = (lookback, n_features)
            
            if model_type == "Bidirectional LSTM":
                model = LSTMModelBuilder._build_bidirectional(
                    input_shape, units_1, units_2, dropout
                )
            elif model_type == "Deep LSTM":
                model = LSTMModelBuilder._build_deep(
                    input_shape, units_1, units_2, dropout
                )
            else:  # Standard LSTM
                model = LSTMModelBuilder._build_standard(
                    input_shape, units_1, units_2, dropout
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
        input_shape: Tuple[int, int],
        units_1: int,
        units_2: int,
        dropout: float
    ) -> Sequential:
        """Build standard LSTM architecture."""
        model = Sequential()
        model.add(LSTM(
            units=units_1,
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units_2, return_sequences=False))
        model.add(Dropout(dropout))
        return model
    
    @staticmethod
    def _build_bidirectional(
        input_shape: Tuple[int, int],
        units_1: int,
        units_2: int,
        dropout: float
    ) -> Sequential:
        """Build bidirectional LSTM architecture."""
        model = Sequential()
        model.add(Bidirectional(
            LSTM(units=units_1, return_sequences=True),
            input_shape=input_shape
        ))
        model.add(Dropout(dropout))
        model.add(Bidirectional(
            LSTM(units=units_2, return_sequences=False)
        ))
        model.add(Dropout(dropout))
        return model
    
    @staticmethod
    def _build_deep(
        input_shape: Tuple[int, int],
        units_1: int,
        units_2: int,
        dropout: float
    ) -> Sequential:
        """Build deep LSTM architecture with 3 LSTM layers."""
        model = Sequential()
        model.add(LSTM(
            units=units_1,
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout))
        model.add(LSTM(units=units_2, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=max(32, units_2 // 2), return_sequences=False))
        model.add(Dropout(dropout))
        return model