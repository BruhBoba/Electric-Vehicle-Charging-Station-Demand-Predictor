# Feature selection using correlation matrices  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Load the dataset
df = pd.read_csv('data/austin_ev.csv')

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

