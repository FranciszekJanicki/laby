import pandas as pd

df = pd.read_csv("titanic.csv")

print(df.groupby('sex')['survived'].mean())
print(df.groupby('pclass')['survived'].mean())