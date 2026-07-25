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

numeric_df = df.select_dtypes(include=np.number)
pearson_corr = numeric_df.corr(method="pearson")
spearman_corr = numeric_df.corr(method="spearman")

target = "utilization_rate"

print("Pearson Correlation: ")
print(
    pearson_corr[target]
    .sort_values(ascending=False)
)

print("\nSpearman Correlation: ")
print(
    spearman_corr[target]
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,8))

sns.heatmap(
    pearson_corr,
    cmap="coolwarm",
    center=0
)

plt.title("Pearson Correlation Matrix")
plt.show()