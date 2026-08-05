import pandas as pd
import os

# 1. Load the massive local dataset
df = pd.read_csv('data/austin_ev.csv')
df = df.sort_values(['station_id', 'timestamp'])
df['util_30_mins_ago'] = df.groupby('station_id')['utilization_rate'].shift(1)
df = df.dropna(subset=['util_30_mins_ago'])

# 2. Calculate the averages
lookup = df.groupby(['hour_of_day', 'location_type']).agg({
    'traffic_congestion_index': 'mean',
    'utilization_rate': 'mean',
    'util_30_mins_ago': 'mean',
    'is_peak_hour': lambda x: x.mode()[0] if not x.mode().empty else 0
}).reset_index()

# 3. Save it as a tiny CSV in the models folder
os.makedirs("models", exist_ok=True)
lookup.to_csv('models/lookup_table.csv', index=False)
print("Tiny lookup table saved successfully!")