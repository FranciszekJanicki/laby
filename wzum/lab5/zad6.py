import pandas as pd

df = pd.read_csv("titanic.csv")
df = df.replace({None: pd.NA})

df['age'] = df['age'].astype(float)
df['age'] = df['age'].fillna(df['age'].median())

df['fare'] = df['fare'].fillna(df['fare'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

df = df.drop(columns=['cabin'])

print(df.isnull().sum())