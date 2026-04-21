import pandas as pd

df = pd.read_csv("titanic.csv")

df = df.replace({None: pd.NA})

print(df.isnull().sum())