"""
Model training utilities.
"""

import streamlit as st
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.models import Sequential
import numpy as np
from typing import Dict, Any, Optional

from app.config import config


class ProgressCallback(Callback):
    """Custom callback for displaying training progress in Streamlit."""
    
    def __init__(self, epochs: int, progress_bar, status_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
    
    def on_epoch_end(self, epoch, logs=None):
        """Update progress bar and status text."""
        progress = min(int((epoch + 1) / self.epochs * 100), 100)
        self.progress_bar.progress(progress)
        
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        self.status_text.text(
            f"Epoch {epoch + 1}/{self.epochs} - "
            f"Loss: {loss:.6f} - "
            f"Val Loss: {val_loss:.6f}"
        )


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
        batch_size: int
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
            
        Returns:
            Training history or None if error
        """
        try:
            # Create callbacks
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LEARNING_RATE,
                verbose=0
            )
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            progress_callback = ProgressCallback(epochs, progress_bar, status_text)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stop, reduce_lr, progress_callback],
                verbose=0
            )
            
            progress_bar.progress(100)
            status_text.success("✅ Pelatihan selesai!")
            
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
