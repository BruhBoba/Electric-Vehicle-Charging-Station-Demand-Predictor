import streamlit as st
import joblib

model = joblib.load(
    "models/random_forest.joblib"
)

st.title("EV Charging Demand Predictor")

st.write("Model loaded successfully!")