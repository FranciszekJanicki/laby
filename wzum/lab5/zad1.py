import pandas as pd

df = pd.read_csv("titanic.csv")

df.info()
df.describe()
df.head()

for col in df.columns:
    print(col, df[col].unique()[:10])