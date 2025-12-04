import numpy as np
import joblib
from tensorflow.keras.models import load_model

def load_prediction_model(model_path):
    # compile=False fixes the optimizer error by skipping the training configuration.
    return load_model(model_path, compile=False)

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def make_prediction(model, scaler_X, scaler_y, input_data, window_size):
    """
    Predicts using separate scalers for Input (X) and Output (y).
    Input data can be multiple columns.
    """
    # 1. Handle dimensions for scaling
    # input_data shape is (window_size, n_features)
    
    # If 1D array (only 1 feature selected), reshape to (window_size, 1)
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 1)
        
    # 2. Scale the Input using scaler_X
    # scaler_X.transform expects (n_samples, n_features)
    # We pass the whole window as a batch of samples to be scaled
    scaled_data = scaler_X.transform(input_data)
    
    # 3. Reshape for LSTM: (samples, time_steps, features)
    # samples=1, time_steps=window_size, features=n_features
    n_features = scaled_data.shape[1]
    final_input = scaled_data.reshape(1, window_size, n_features)
    
    # 4. Predict (returns scaled value)
    prediction_scaled = model.predict(final_input)
    
    # 5. Inverse Scale the Output using scaler_y
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction[0][0]