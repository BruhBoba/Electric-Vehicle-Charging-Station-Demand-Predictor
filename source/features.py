import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/austin_ev.csv')



# splitting data into train/test sets
y = df['utili']