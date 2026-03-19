import streamlit as st
import numpy as np
import joblib

# Load model bundle
bundle = joblib.load("gold_price_prediction.pkl")
model = bundle["model"]
x_scaler = bundle["x_scaler"]
y_scaler = bundle["y_scaler"]

st.set_page_config(page_title="Gold Price Prediction", layout="centered")

st.title("🪙 Gold Price Prediction using ANN")
st.write("Prediction based on Open, High, Low, Volume")

# ---- INPUT FEATURES ----
open_price = st.number_input("Open Price", value=2000.0)
high_price = st.number_input("High Price", value=2020.0)
low_price = st.number_input("Low Price", value=1980.0)
volume = st.number_input("Volume", value=100000.0)

# Order MUST match training
input_data = np.array([[open_price, high_price, low_price, volume]])

if st.button("Predict Gold Price"):
    input_scaled = x_scaler.transform(input_data)
    pred_scaled = model.predict(input_scaled)
    prediction = y_scaler.inverse_transform(pred_scaled)

    st.success(f"💰 Predicted Gold Closing Price: {prediction[0][0]:.2f}")
