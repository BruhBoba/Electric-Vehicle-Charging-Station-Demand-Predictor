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

# drop first row, cuz util rate is NaN atp
df = df.dropna(subset=['util_30_mins_ago'])

# assign x and y vals
y = df['utilization_rate']
X = df.drop(columns=['utilization_rate', 'timestamp'])

# split data into 80/20 train/test
# REMOVE RANDOM STATE PARAM B4 DEPLOYING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=117)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")



plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


# Remove features that cause leakage or are not useful
drop_cols = [
    "timestamp", 
    "station_id",
    "station_name",
    "city",
    "state",                
    "ports_total",
    "ports_available",
    "ports_occupied",
    "ports_out_of_service",
    "estimated_wait_time_mins",
    "current_price",
    "pricing_type"
]

df_model = df.drop(columns=drop_cols)
print(df_model.columns)

# Select only numeric columns
numeric_df = df_model.select_dtypes(include=["int64", "float64", "bool"])

# Pearson correlation
pearson = numeric_df.corr(method="pearson")
# Sort correlations with utilization_rate
pearson_target = pearson["utilization_rate"].sort_values(ascending=False)

# Spearman correlation
spearman = numeric_df.corr(method="spearman")
spearman_target = spearman["utilization_rate"].sort_values(ascending=False)

# Compare both in a table 
correlation_df = pd.DataFrame({
    "Pearson": pearson_target,
    "Spearman": spearman_target
})

print(correlation_df)

