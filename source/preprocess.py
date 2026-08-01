import pandas as pd

# Filters the dataset to where city == "Austin"
df = pd.read_csv("data/ev_charging_station_data.csv")

austin = df[df["city"] == "Austin"]

austin.to_csv("data/austin_ev.csv", index=False)

print("Austin dataset saved.")