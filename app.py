import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/random_forest.joblib")
features = joblib.load('models/features.joblib')

st.title("EV Charging Demand Predictor")
st.write("Predict station utilization 30 minutes in advance!")

# Create input widgets for user interaction
hour = st.slider("Hour of Day", 0, 23, 14)
traffic = st.slider("Traffic Congestion Index", 0.0, 1.0, 0.5)
current_util = st.slider("Current Utilization Rate", 0.0, 1.0, 0.4)
util_30_ago = st.slider("Utilization 30 Mins Ago", 0.0, 1.0, 0.35)

# Build input dataframe matching exact model features
input_data = pd.DataFrame(0, index=[0], columns=features)
input_data['hour_of_day'] = hour
input_data['traffic_congestion_index'] = traffic
input_data['utilization_rate'] = current_util
input_data['util_30_mins_ago'] = util_30_ago

if st.button("Predict 30-Min Demand"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Utilization: {prediction * 100:.1f}%")