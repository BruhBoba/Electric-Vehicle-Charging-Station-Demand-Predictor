# This is the preprocessing and EDA for the data
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
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f'Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'Date Range: {df["timestamp"].min()} to {df["timestamp"].max()}')

# Check data types
df.info()

# Missing values analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage': missing_pct})
print('Missing Values Summary:')
print(missing_df[missing_df['Missing Count'] > 0] if missing.sum() > 0 else 'No missing values found')

def get_summary(df, cols):
    return df[cols].describe()

def check_imbalance(df, col):
    return df[col].value_counts(normalize=True)

def find_extreme_values(df, col):
    return df[df[col] > df[col].quantile(0.99)]

# Check duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates:,}")
# Remove duplicates if found
df = df.drop_duplicates()
print(f"Dataset after removing duplicates: {df.shape}")

# Checks if utilization rate is within range 
print(df["utilization_rate"].min())
print(df["utilization_rate"].max())

# Checks the distribution of utilization rate
print(df["utilization_rate"].describe())

# Check unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print("\n", col)
    print(df[col].unique())


features = []