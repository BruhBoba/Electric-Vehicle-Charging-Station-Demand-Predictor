# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


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
# REMOVE RANDOM STATE PARAM B4 DEPLOYING
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=117)

# properly encodes amenities feature which has various elements in one long string
amenities_encoded = x_train['amenities'].str.get_dumies(sep=', ')
x_train  = x_train.drop(columns=['amenities'])
x_train = pd.concat([x_train, amenities_encoded], axis = 1)

# encodes other non-numeric values
x_train_encoded = pd.get_dummies(x_train)


# pearson correlation: checks 4 linear relationships
pearson = numeric_df.corr(method="pearson")
# Sort correlations with utilization_rate
pearson_target = pearson["utilization_rate"].sort_values(ascending=False)

# spearman correlation: checks for any trends (monotonic)
spearman = numeric_df.corr(method="spearman")
spearman_target = spearman["utilization_rate"].sort_values(ascending=False)

# Compare both in a table 
correlation_df = pd.DataFrame({
    "Pearson": pearson_target,
    "Spearman": spearman_target
})

print(correlation_df)

