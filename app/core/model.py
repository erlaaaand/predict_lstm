import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import traceback

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
            # print(f"DEBUG: Predicting day {i+1}")
            pred = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
            predictions.append(pred[0, 0])
            current_sequence = np.append(current_sequence[1:], pred[0, 0])
        
        print("DEBUG: Inverse transforming predictions")
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        print("DEBUG: Prediction complete")
        return predictions.flatten()
        
    except Exception as e:
        print(f"DEBUG ERROR in predict_future: {str(e)}")
        # import traceback
        # traceback.print_exc()
        st.error(f"Error in prediction: {str(e)}")
        return None
