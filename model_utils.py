import numpy as np
import joblib
from tensorflow.keras.models import load_model

def load_prediction_model(model_path):
    """Loads the saved Keras model."""
    return load_model(model_path)

def load_scaler(scaler_path):
    """Loads the saved Scikit-Learn scaler (StandardScaler or MinMaxScaler)."""
    return joblib.load(scaler_path)

def make_prediction(model, scaler, input_data, window_size):
    """
    Preprocesses data and makes a prediction.
    
    Args:
        model: Loaded Keras model.
        scaler: Loaded Scaler.
        input_data: Numpy array of shape (window_size,).
        window_size: The lookback period expected by the model.
        
    Returns:
        float: The de-scaled prediction value.
    """
    # 1. Reshape to (n_samples, 1) for the scaler
    input_data_reshaped = input_data.reshape(-1, 1)
    
    # 2. Scale the data (transform)
    # Note: We assume the scaler was fit on this single feature during training
    scaled_data = scaler.transform(input_data_reshaped)
    
    # 3. Reshape for LSTM: (samples, time_steps, features)
    # samples=1, time_steps=window_size, features=1
    final_input = scaled_data.reshape(1, window_size, 1)
    
    # 4. Predict
    prediction_scaled = model.predict(final_input)
    
    # 5. Inverse Scale (convert back to original units)
    prediction = scaler.inverse_transform(prediction_scaled)
    
    return prediction[0][0]