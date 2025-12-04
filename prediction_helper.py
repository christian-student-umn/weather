import numpy as np
import joblib
from tensorflow.keras.models import load_model

def load_prediction_model(model_path):
    return load_model(model_path)

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def make_prediction(model, scaler_X, scaler_y, input_data, window_size):
    """
    Predicts using separate scalers for Input (X) and Output (y).
    """
    # 1. Reshape Input for Scaling: (n_samples, n_features)
    input_data_reshaped = input_data.reshape(-1, 1)
    
    # 2. Scale the Input using scaler_X
    scaled_data = scaler_X.transform(input_data_reshaped)
    
    # 3. Reshape for LSTM: (samples, time_steps, features)
    final_input = scaled_data.reshape(1, window_size, 1)
    
    # 4. Predict (returns scaled value)
    prediction_scaled = model.predict(final_input)
    
    # 5. Inverse Scale the Output using scaler_y
    # The model output shape is typically (1, 1)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    return prediction[0][0]