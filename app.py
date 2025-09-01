# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load("fitness_model.pkl")

st.set_page_config(page_title="Body Performance Classifier", layout="centered")

st.title("🏋️ Body Performance Classifier")
st.write("Enter the details below to predict the body performance class (A, B, C, D).")

# User inputs
gender = st.selectbox("Gender", ["M", "F"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
body_fat = st.number_input("Body Fat (%)", min_value=2.0, max_value=60.0, value=20.0)
diastolic = st.number_input("Diastolic (mmHg)", min_value=40, max_value=150, value=80)
systolic = st.number_input("Systolic (mmHg)", min_value=70, max_value=250, value=120)
gripForce = st.number_input("Grip Force", min_value=10, max_value=100, value=40)
sit_bend = st.number_input("Sit and Bend Forward (cm)", min_value=-20, max_value=50, value=5)
situps = st.number_input("Sit-ups Count", min_value=0, max_value=100, value=30)
broad_jump = st.number_input("Broad Jump (cm)", min_value=50, max_value=300, value=150)

# Predict button
if st.button("Predict"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "body fat_%": body_fat,
        "diastolic": diastolic,
        "systolic": systolic,
        "gripForce": gripForce,
        "sit and bend forward_cm": sit_bend,
        "sit-ups counts": situps,
        "broad jump_cm": broad_jump
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.success(f"### 🏆 Predicted Class: **{prediction}**")
    st.write("#### Class Probabilities:")
    for cls, prob in zip(model.classes_, probabilities):
        st.write(f"- {cls}: {prob:.2%}")
