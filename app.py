import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_utils import load_prediction_model, load_scaler, make_prediction

# --- Page Config ---
st.set_page_config(
    page_title="Weather/Temperature Forecaster",
    page_icon="g",
    layout="wide"
)

# --- Title and Description ---
st.title("g Weather Forecasting App (LSTM)")
st.markdown("""
This application uses a Deep Learning (LSTM) model to forecast future temperatures.
**Upload a CSV file containing the past weather data to generate a forecast.**
""")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")
window_size = st.sidebar.number_input(
    "Lookback Window Size", 
    min_value=1, 
    value=60, 
    help="The number of past days/hours the model needs to predict the next step. (Match this to your training code!)"
)

# --- Load Model & Scaler ---
try:
    model = load_prediction_model('model.keras')
    scaler = load_scaler('scaler.pkl')
    st.sidebar.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.info("Please ensure 'model.keras' and 'scaler.pkl' are in the same directory.")
    st.stop()

# --- Main Interface ---
uploaded_file = st.file_uploader("Upload your input CSV (must contain a 'Temp' or relevant column)", type=["csv"])

if uploaded_file is not None:
    # Read Data
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Column Selection
    target_col = st.selectbox("Select the column to use for prediction", df.columns)

    # Check if we have enough data
    if len(df) < window_size:
        st.error(f"Not enough data! The model needs at least {window_size} rows to make the first prediction.")
    else:
        if st.button("Generate Forecast"):
            with st.spinner("Calculating forecast..."):
                try:
                    # Prepare data (Taking the last 'window_size' points)
                    input_data = df[target_col].values[-window_size:]
                    
                    # Make Prediction
                    prediction = make_prediction(model, scaler, input_data, window_size)
                    
                    # Display Result
                    st.success("Prediction Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Predicted Next Value", value=f"{prediction:.4f}")
                    
                    # Visualization
                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        # Plot historical context (last 50 points) + prediction
                        history_plot = df[target_col].values[-50:]
                        x_hist = np.arange(len(history_plot))
                        x_pred = len(history_plot)
                        
                        ax.plot(x_hist, history_plot, label='History', marker='o')
                        ax.scatter(x_pred, prediction, color='red', label='Forecast', zorder=5, s=100)
                        ax.set_title("Forecast Visualization")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Awaiting CSV file upload. Please upload a file to proceed.")