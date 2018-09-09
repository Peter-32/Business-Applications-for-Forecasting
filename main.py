# Imports
import pandas as pd

# Read in data
df = pd.read_csv('data/train_1.csv', sep=",", nrows=1)
df = df.T
df.columns = df.iloc[0]
df = df.iloc[1:]
