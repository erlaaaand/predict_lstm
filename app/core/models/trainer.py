"""
Model training utilities.
Fixed version.
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
import numpy as np
from typing import Dict, Any, Optional


class ModelTrainer:
    """Handles model training process."""
    
    @staticmethod
    def train(
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: int,
        callbacks: list = None
    ) -> Optional[tf.keras.callbacks.History]:
        """
        Train LSTM model with callbacks.
        
        Args:
            model: Compiled Keras model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks: List of callbacks
            
        Returns:
            Training history or None if error
        """
        try:
            if callbacks is None:
                callbacks = []
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            return history
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            return None
    
    @staticmethod
    def get_training_summary(history: tf.keras.callbacks.History) -> Dict[str, Any]:
        """
        Get summary of training metrics.
        
        Args:
            history: Training history
            
        Returns:
            Dictionary with training summary
        """
        return {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['loss']),
            'history': history.history
        }