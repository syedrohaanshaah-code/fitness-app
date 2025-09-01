# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Fitness App", layout="wide")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "fitness_model.pkl")
model = joblib.load(model_path)

st.title("🏋️ Fitness Performance Classifier")

# Example inputs (adjust according to your dataset features)
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
gender = st.selectbox("Gender", ["M", "F"])

if st.button("Predict"):
    # Create dataframe for model
    input_data = pd.DataFrame([{
        "age": age,
        "height": height,
        "weight": weight,
        "gender": gender
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Performance: {prediction}")
