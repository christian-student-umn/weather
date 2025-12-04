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
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.error("❌ No numeric columns found!")
            st.stop()
            
        st.write("### Data Preview")
        st.dataframe(df.head())

        # --- NEW: Feature Selection ---
        st.subheader("Model Inputs")
        st.info("Your model was trained on multiple features. Please select them all in the correct order.")
        
        # 1. Feature Columns (Input for the Model)
        feature_cols = st.multiselect(
            "Select Input Features (Must match training data)", 
            numeric_cols,
            default=numeric_cols  # Select all by default to be helpful
        )
        
        # 2. Target Column (For Visualization Only)
        target_col = st.selectbox(
            "Select Target Column (For Plotting)", 
            numeric_cols,
            index=0
        )

        # Check data length
        if len(df) < window_size:
            st.error(f"Not enough data! The model needs at least {window_size} rows.")
        else:
            if st.button("Generate Forecast"):
                if not feature_cols:
                    st.error("Please select at least one feature column.")
                else:
                    with st.spinner("Calculating forecast..."):
                        try:
                            # Prepare data: Get last 'window_size' rows of the SELECTED FEATURES
                            input_data = df[feature_cols].values[-window_size:]
                            
                            # Make Prediction
                            prediction = make_prediction(model, scaler_X, scaler_y, input_data, window_size)
                            
                            # Display Result
                            st.success("Prediction Complete!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(label="Predicted Next Value", value=f"{prediction:.4f}")
                            
                            # Visualization
                            with col2:
                                fig, ax = plt.subplots(figsize=(10, 4))
                                # Plot historical context of the TARGET column
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
                            st.warning(f"Your model expects specific input dimensions. You selected {len(feature_cols)} features. If you trained on 9 features, you must select 9 here.")
    
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")

else:
    st.info("Awaiting CSV file upload.")