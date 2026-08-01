# import libraries 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# load dataset
df = pd.read_csv('data/austin_ev.csv') 

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['station_id', 'timestamp'])
# new col to hold past and future timestamp's util rate
# This groups the station before shifting 
df['util_30_mins_ago'] = df.groupby('station_id')['utilization_rate'].shift(1)
df['target_util_30_ahead'] = df.groupby('station_id') ['utilization_rate'].shift(-1)

# drops the first row of each station because new col is empty 
df = df.dropna(subset=['util_30_mins_ago', 'target_util_30_ahead'])

# Confirms if NaN is each station's first row is gone 
print(df.head())

# remove features that cause leakage or unnecessary
drop_cols = [
    "timestamp", "station_id", "station_name", "city", "state",                 
    "ports_total", "ports_available", "ports_occupied", 
    "ports_out_of_service", "estimated_wait_time_mins", 
    "current_price", "pricing_type", "station_status", 
    "latitude", "longitude", "avg_session_duration_mins"
]

# make new df
df_model = df.drop(columns=drop_cols)

# assign x and y vals
y = df_model['target_util_30_ahead']
x = df_model.drop(columns=['target_util_30_ahead'])

# split data by time series so train data is from Jul-Oct and test data is from Nov-Dec 
df['timestamp'] = pd.to_datetime(df['timestamp']) 
cutoff = pd.to_datetime('2025-11-01') 
train_mask = df['timestamp'] <  cutoff       # Jul–Oct trains 
test_mask  = df['timestamp'] >= cutoff       # Nov–Dec tests 

x_train, x_test = x[train_mask], x[test_mask] 
y_train, y_test = y[train_mask], y[test_mask]

# properly encodes amenities feature which has various amenities in one long string
amenities_encoded_train = x_train['amenities_nearby'].str.get_dummies(sep=', ')
x_train  = x_train.drop(columns=['amenities_nearby'])
x_train = pd.concat([x_train, amenities_encoded_train], axis = 1)

amenities_encoded_test = x_test['amenities_nearby'].str.get_dummies(sep=', ')
x_test = x_test.drop(columns=['amenities_nearby'])
x_test = pd.concat([x_test, amenities_encoded_test], axis=1)


# encodes other non-numeric values
x_train_encoded = pd.get_dummies(x_train)
x_test_encoded = pd.get_dummies(x_test)

# Align columns of both testing and training data 
x_train_encoded, x_test_encoded = x_train_encoded.align(
    x_test_encoded,
    join="left",
    axis=1,
    fill_value=0
)

correlation_data = x_train_encoded.copy()
correlation_data['target_util_30_ahead'] = y_train

# pearson correlation: checks for linear relationships
pearson = correlation_data.corr(method="pearson")
pearson_target = pearson["target_util_30_ahead"].sort_values(ascending=False)

# spearman correlation: checks for any trends (monotonic)
spearman = correlation_data.corr(method="spearman")
spearman_target = spearman["target_util_30_ahead"].sort_values(ascending=False)

print("\nPearson")
print(pearson_target)

print("\nSpearman")
print(spearman_target)

# results are summarized below
'''
Strong Correlations (Absolute Value >= 0.2)
    utilization_rate: Spearman (0.938574) | Pearson (0.927771)
    util_30_mins_ago: Spearman (0.916336) | Pearson (0.903775)
    traffic_congestion_index: Spearman (0.426122) | Pearson (0.404035)
    hour_of_day: Spearman (0.406061) | Pearson (0.397594)
    is_peak_hour: Spearman (0.254299) | Pearson (0.263299)
    location_type_Shopping Center: Spearman (0.220405) | Pearson (0.246511)
    Restaurant: Spearman (-0.231940) | Pearson (-0.164509)
    location_type_Workplace: Spearman (-0.231940) | Pearson (-0.164509)
    Convenience Store: Spearman (-0.285423) | Pearson (-0.277442)

List 2: Weaker Correlations (Absolute Value < 0.2)
    Grocery Store: Spearman (0.176244) | Pearson (0.155210)
    charger_type_DC Fast Charge: Spearman (0.174196) | Pearson (0.153162)
    network_ChargePoint: Spearman (-0.117630) | Pearson (-0.175286)
    charger_type_Level 2: Spearman (-0.117630) | Pearson (-0.175286)
    location_type_Residential: Spearman (-0.117630) | Pearson (-0.175286)
    Park: Spearman (0.136224) | Pearson (0.152211)
    Hotel: Spearman (0.133716) | Pearson (0.149702)
    network_Blink: Spearman (0.133716) | Pearson (0.149702)
    Shopping Mall: Spearman (-0.133716) | Pearson (-0.149702)
    WiFi: Spearman (0.124361) | Pearson (0.103391)
    Restroom: Spearman (0.031027) | Pearson (0.112190)
    power_output_kw: Spearman (0.089339) | Pearson (0.066334)
    network_Shell Recharge: Spearman (0.079630) | Pearson (0.037882)
    location_type_Hotel/Hospitality: Spearman (0.079630) | Pearson (0.037882)
    network_Tesla Supercharger: Spearman (-0.078152) | Pearson (-0.010042)
    charger_type_Tesla DC Fast: Spearman (-0.078152) | Pearson (-0.010042)
    is_weekend: Spearman (-0.065019) | Pearson (-0.059241)
    day_of_week: Spearman (-0.052530) | Pearson (-0.048485)
    month: Spearman (-0.020057) | Pearson (-0.020409)
    Coffee Shop: Spearman (0.013135) | Pearson (-0.020889)
    Fast Food: Spearman (0.013135) | Pearson (-0.020889)
    temperature_f: Spearman (0.015149) | Pearson (0.016717)
    gas_price_per_gallon: Spearman (0.009325) | Pearson (0.009575)
    weather_condition_cloudy: Spearman (0.008268) | Pearson (0.008201)
    local_event_sports_game: Spearman (0.005417) | Pearson (0.006528)
    local_event_conference: Spearman (0.005625) | Pearson (0.004794)
    weather_condition_partly_cloudy: Spearman (0.004497) | Pearson (0.004120)
    precipitation_mm: Spearman (0.001671) | Pearson (0.002670)
    local_event_festival: Spearman (0.001289) | Pearson (0.002196)
    weather_condition_light_rain: Spearman (0.001185) | Pearson (0.001552)
    weather_condition_heavy_rain: Spearman (0.000489) | Pearson (0.000074)
    local_event_none: Spearman (-0.000746) | Pearson (-0.001218)
    weather_condition_extreme_heat: Spearman (-0.001326) | Pearson (-0.001084)
    weather_condition_clear: Spearman (-0.009478) | Pearson (-0.009512)
    local_event_concert: Spearman (-0.011602) | Pearson (-0.011814)
'''

# decided based on comment above
features_to_keep = [
    'hour_of_day', 
    'traffic_congestion_index',
    'utilization_rate', 
    'util_30_mins_ago', 
    'is_peak_hour', 
    'Convenience Store', 
    'Restaurant', 
    'Grocery Store', 
    'Hotel', 
    'Shopping Mall', 
    'Park', 
    'WiFi', 
    'Restroom',
    'location_type_Shopping Center', 
    'location_type_Workplace', 
    'location_type_Residential',
    'charger_type_DC Fast Charge', 
    'charger_type_Level 2',
    'network_ChargePoint', 
    'network_Blink'
]

# create final dfs that hold the features we want to KEEP
x_train_final = x_train_encoded[features_to_keep]
x_test_final = x_test_encoded[features_to_keep]

def prepare_data():
    return (
        x_train_final,
        x_test_final,
        y_train,
        y_test
    )