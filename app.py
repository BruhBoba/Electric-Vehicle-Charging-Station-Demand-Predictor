import streamlit as st
import joblib
import pandas as pd

model = joblib.load("models/random_forest.joblib")
features = joblib.load('models/features.joblib')

st.title("EV Charging Demand Predictor Model")
st.write("Predict Austin's EV station utilization rate 30 minutes in advance! Our model analyzes a synnthetic dataset composed of data such as " \
"traffic, the type of amenities surrounding that station, and the EV network. Our model aims to help EV owners make data-driven decisions " \
"and charge their cars when demand is not experiancing a spike." )

# Cache the dataset load so it only runs once when the app starts
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv('data/austin_ev.csv')
    df = df.sort_values(['station_id', 'timestamp'])
    df['util_30_mins_ago'] = df.groupby('station_id')['utilization_rate'].shift(1)
    df = df.dropna(subset=['util_30_mins_ago'])
    
    # Create a lookup table of historical averages grouped by location type and hour
    lookup = df.groupby(['hour_of_day', 'location_type']).agg({
        'traffic_congestion_index': 'mean',
        'utilization_rate': 'mean',
        'util_30_mins_ago': 'mean',
        'is_peak_hour': lambda x: x.mode()[0] if not x.mode().empty else 0
    }).reset_index()
    
    return df, lookup

df, lookup = load_and_prep_data()

# Driver-focused UI Inputs
location_options = df['location_type'].dropna().unique()
selected_location = st.selectbox("Destination Type", location_options)
arrival_hour = st.slider("Intended Arrival Time (Hour of Day)", 0, 23, 12)

if st.button("Check Availability"):
    # 1. Fetch historical background data for the specific hour and location
    baseline = lookup[(lookup['hour_of_day'] == arrival_hour) & (lookup['location_type'] == selected_location)]
    
    if not baseline.empty:
        traffic = baseline['traffic_congestion_index'].values[0]
        current_util = baseline['utilization_rate'].values[0]
        util_30_ago = baseline['util_30_mins_ago'].values[0]
        peak = baseline['is_peak_hour'].values[0]
    else:
        # Fallback to global averages if the specific combination is missing
        traffic = df['traffic_congestion_index'].mean()
        current_util = df['utilization_rate'].mean()
        util_30_ago = df['util_30_mins_ago'].mean()
        peak = 1 if (16 <= arrival_hour <= 19) else 0

    # 2. Build the exact dataframe the Random Forest model expects, initialized with 0s
    input_data = pd.DataFrame(0, index=[0], columns=features)
    
    # 3. Populate the known variables
    if 'hour_of_day' in input_data.columns: input_data['hour_of_day'] = arrival_hour
    if 'traffic_congestion_index' in input_data.columns: input_data['traffic_congestion_index'] = traffic
    if 'utilization_rate' in input_data.columns: input_data['utilization_rate'] = current_util
    if 'util_30_mins_ago' in input_data.columns: input_data['util_30_mins_ago'] = util_30_ago
    if 'is_peak_hour' in input_data.columns: input_data['is_peak_hour'] = peak
    
    # Enable the specific one-hot encoded location column if it exists in the training features
    loc_col = f'location_type_{selected_location}'
    if loc_col in input_data.columns:
        input_data[loc_col] = 1

    # 4. Generate prediction
    prediction = model.predict(input_data)[0]
    
    # 5. Display driver-friendly output
    st.write("---")
    st.write(f"### Forecasted Station Utilization: {prediction * 100:.1f}%")
    
    if prediction >= 0.80:
        st.error("🚨 High Demand Expected. Chargers will likely be full. Consider alternate times or locations.")
    elif prediction >= 0.50:
        st.warning("⚠️ Moderate Demand. You may experience a short wait upon arrival.")
    else:
        st.success("✅ Chargers Likely Available. Good time to charge!")

