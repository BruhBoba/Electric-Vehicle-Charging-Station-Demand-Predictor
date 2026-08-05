import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/random_forest.joblib")
features = joblib.load('models/features.joblib')

st.title("EV Charging Demand Predictor Model")
st.write("Predict Austin's EV station utilization rate 30 minutes in advance! Our model analyzes a synnthetic dataset composed of data such as " \
"traffic, the type of amenities surrounding that station, and the EV network. Our model aims to help EV owners make data-driven decisions " \
"and charge their cars when demand is not experiancing a spike." )

model = joblib.load("models/random_forest.joblib")
features = joblib.load("models/features.joblib")
lookup = pd.read_csv("models/lookup_table.csv")


# Driver-focused UI Inputs
location_options = lookup['location_type'].dropna().unique()
selected_location = st.selectbox("Destination Type", location_options)
arrival_hour = st.slider("Intended Arrival Time (Hour of Day)", 0, 23, 12)

if st.button("Check Availability"):
    # Fetch historical background data from the tiny lookup table
    baseline = lookup[(lookup['hour_of_day'] == arrival_hour) & (lookup['location_type'] == selected_location)]
    
    if not baseline.empty:
        traffic = baseline['traffic_congestion_index'].values[0]
        current_util = baseline['utilization_rate'].values[0]
        util_30_ago = baseline['util_30_mins_ago'].values[0]
        peak = baseline['is_peak_hour'].values[0]
    else:
        # Fallbacks if exact combination doesn't exist
        traffic = lookup['traffic_congestion_index'].mean()
        current_util = lookup['utilization_rate'].mean()
        util_30_ago = lookup['util_30_mins_ago'].mean()
        peak = 1 if (16 <= arrival_hour <= 19) else 0

    input_data = pd.DataFrame(0, index=[0], columns=features)
    
    if 'hour_of_day' in input_data.columns: input_data['hour_of_day'] = arrival_hour
    if 'traffic_congestion_index' in input_data.columns: input_data['traffic_congestion_index'] = traffic
    if 'utilization_rate' in input_data.columns: input_data['utilization_rate'] = current_util
    if 'util_30_mins_ago' in input_data.columns: input_data['util_30_mins_ago'] = util_30_ago
    if 'is_peak_hour' in input_data.columns: input_data['is_peak_hour'] = peak
    
    loc_col = f'location_type_{selected_location}'
    if loc_col in input_data.columns:
        input_data[loc_col] = 1

    prediction = model.predict(input_data)[0]
    
    st.write("---")
    st.write(f"### Forecasted Station Utilization: {prediction * 100:.1f}%")
    
    if prediction >= 0.80:
        st.error("🚨 High Demand Expected. Chargers will likely be full.")
    elif prediction >= 0.50:
        st.warning("⚠️ Moderate Demand. You may experience a short wait.")
    else:
        st.success("✅ Chargers Likely Available. Good time to charge!")