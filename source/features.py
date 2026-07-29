# import libraries 
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# load dataset
df = pd.read_csv('data/austin_ev.csv') 
df = df.sort_values(by='timestamp')

# util = utilization rate
# new col to hold past timestamp's util rate
df['util_30_mins_ago'] = df['utilization_rate'].shift(1)

# remove first row cuz new col is empty atp
df = df.dropna(subset=['util_30_mins_ago'])

# remove features that cause leakage or r unnecessary
drop_cols = [
    "timestamp", "station_id", "station_name", "city", "state",                 
    "ports_total", "ports_available", "ports_occupied", 
    "ports_out_of_service", "estimated_wait_time_mins", 
    "current_price", "pricing_type", "station_status", 
    "latitude", "longitude"
]

# make new df
df_model = df.drop(columns=drop_cols)

# assign x and y vals
y = df_model['utilization_rate']
x = df_model.drop(columns=['utilization_rate'])


# split data into 80/20 train/test
#--------TO DO: REMOVE RANDOM STATE PARAM B4 DEPLOYING--------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=117)

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



correlation_data = x_train_encoded.copy()
correlation_data['utilization_rate'] = y_train

# pearson correlation: checks 4 linear relationships
pearson = correlation_data.corr(method="pearson")
pearson_target = pearson["utilization_rate"].sort_values(ascending=False)

# spearman correlation: checks for any trends (monotonic)
spearman = correlation_data.corr(method="spearman")
spearman_target = spearman["utilization_rate"].sort_values(ascending=False)


# results are summarized below
'''
Strong Correlations (Absolute Value >= 0.2)
    hour_of_day: Spearman (0.452600) | Pearson (0.453762)
    traffic_congestion_index: Spearman (0.408028) | Pearson (0.378107)
    util_30_mins_ago: Spearman (0.359966) | Pearson (0.368518)
    Convenience Store: Spearman (-0.292414) | Pearson (-0.286200)
    location_type_Shopping Center: Spearman (0.225600) | Pearson (0.251434)
    Restaurant: Spearman (-0.239310) | Pearson (-0.174610)
    location_type_Workplace: Spearman (-0.239310) | Pearson (-0.174610)
    is_peak_hour: Spearman (0.229967) | Pearson (0.229136)

List 2: Weaker Correlations (Absolute Value < 0.2)
    charger_type_DC Fast Charge: Spearman (0.181741) | Pearson (0.162964)
    Grocery Store: Spearman (0.177667) | Pearson (0.158213)
    network_ChargePoint: Spearman (-0.118603) | Pearson (-0.175849)
    charger_type_Level 2: Spearman (-0.118603) | Pearson (-0.175849)
    location_type_Residential: Spearman (-0.118603) | Pearson (-0.175849)
    avg_session_duration_mins: Spearman (-0.086457) | Pearson (-0.167384)
    network_Blink: Spearman (0.140918) | Pearson (0.157199)
    Hotel: Spearman (0.140918) | Pearson (0.157199)
    Shopping Mall: Spearman (-0.140918) | Pearson (-0.157199)
    Park: Spearman (0.135676) | Pearson (0.151069)
    WiFi: Spearman (0.128533) | Pearson (0.107670)
    Restroom: Spearman (0.029688) | Pearson (0.108376)
    power_output_kw: Spearman (0.087904) | Pearson (0.063340)
    network_Shell Recharge: Spearman (0.081864) | Pearson (0.042758)
    charger_type_Tesla DC Fast: Spearman (-0.084968) | Pearson (-0.019510)
    network_Tesla Supercharger: Spearman (-0.084968) | Pearson (-0.019510)
    location_type_Hotel/Hospitality: Spearman (0.081864) | Pearson (0.042758)
    is_weekend: Spearman (-0.067602) | Pearson (-0.064354)
    day_of_week: Spearman (-0.054446) | Pearson (-0.052835)
    gas_price_per_gallon: Spearman (0.037365) | Pearson (0.038277)
    month: Spearman (0.028624) | Pearson (0.029390)
    weather_condition_heavy_rain: Spearman (-0.026641) | Pearson (-0.024995)
    weather_condition_extreme_heat: Spearman (0.026303) | Pearson (0.025333)
    precipitation_mm: Spearman (-0.012773) | Pearson (-0.018167)
    Fast Food: Spearman (0.017960) | Pearson (-0.015517)
    Coffee Shop: Spearman (0.017960) | Pearson (-0.015517)
    temperature_f: Spearman (-0.013142) | Pearson (-0.017217)
    weather_condition_partly_cloudy: Spearman (-0.010948) | Pearson (-0.010068)
    weather_condition_clear: Spearman (-0.007721) | Pearson (-0.008317)
    local_event_sports_game: Spearman (0.003660) | Pearson (0.005319)
    local_event_concert: Spearman (-0.004824) | Pearson (-0.004315)
    weather_condition_light_rain: Spearman (-0.002195) | Pearson (-0.001342)
    weather_condition_cloudy: Spearman (0.001668) | Pearson (0.001404)
    local_event_none: Spearman (0.001472) | Pearson (-0.000311)
    local_event_festival: Spearman (-0.001301) | Pearson (0.000301)
    local_event_conference: Spearman (-0.000679) | Pearson (-0.000905)
'''

# decided based on comment above
features_to_keep = [
    'hour_of_day', 
    'traffic_congestion_index', 
    'util_30_mins_ago', 
    'is_peak_hour', 
    'avg_session_duration_mins',
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

# create final dfs that hold the features we want to KEEP, lowk couldve js dropped from old ones too
x_train_final = x_train_encoded[features_to_keep]
x_test_final = x_test_encoded[features_to_keep]

# init model
rf_model = RandomForestRegressor(random_state=117)

# train model
rf_model.fit(x_train_final, y_train)

y_prediction = rf_model.predict(x_test_final)
mae = mean_absolute_error(y_test, y_prediction)
# the function below is resulting in errors, lowk tis discontinued
#rmse = mean_squared_error(y_test, y_prediction, squared=False) 
r2 = r2_score(y_test, y_prediction)

print("--- MODEL REPORT CARD ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")  #0.0753
#print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2): {r2:.4f}")      #0.8534