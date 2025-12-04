import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prediction_helper import load_prediction_model, load_scaler, make_prediction

# --- Page Config ---
st.set_page_config(
    page_title="Weather Forecaster",
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
    "Lookback Window Size (Time Steps)", 
    min_value=1, 
    value=24, 
    help="The number of past steps the model needs to predict the next one."
)

# --- Load Model & Scalers ---
try:
    model = load_prediction_model('model.keras')
    scaler_X = load_scaler('scaler_X.pkl')
    scaler_y = load_scaler('scaler_y.pkl')
    st.sidebar.success("Model and Scalers loaded!")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.info("Please ensure 'model.keras', 'scaler_X.pkl', and 'scaler_y.pkl' are in the repository.")
    st.stop()

# --- Main Interface ---
uploaded_file = st.file_uploader("Upload your input CSV", type=["csv"])

if uploaded_file is not None:
    # Read Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- ROBUSTNESS FIX ---
        # 1. Check if the file was read correctly (more than 1 column?)
        # 2. Filter ONLY numeric columns for the prediction target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ No numeric columns found!")
            st.warning("""
            The app could not find any numbers in your CSV. 
            This often happens if the CSV is read as a single line of text.
            
            **Troubleshooting:**
            1. Open your CSV file and ensure columns are separated by commas.
            2. Ensure you didn't save it with a weird encoding.
            """)
            st.write("### How pandas sees your data (Preview):")
            st.dataframe(df.head())
            st.stop()
            
        st.write("### Data Preview")
        st.dataframe(df.head())

        # Column Selection: Only show numeric columns to prevent string errors
        target_col = st.selectbox("Select the column to use for prediction", numeric_cols)

        # Check data length
        if len(df) < window_size:
            st.error(f"Not enough data! The model needs at least {window_size} rows.")
        else:
            if st.button("Generate Forecast"):
                with st.spinner("Calculating forecast..."):
                    try:
                        # Prepare data (Taking the last 'window_size' points)
                        input_data = df[target_col].values[-window_size:]
                        
                        # Make Prediction using BOTH scalers
                        prediction = make_prediction(model, scaler_X, scaler_y, input_data, window_size)
                        
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
                        st.error(f"An error occurred: {e}")
                        st.warning("If this fails, check if your model expects more than 1 feature (column).")
    
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")

else:
    st.info("Awaiting CSV file upload.")