import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("titanic.csv")
df = df.replace({None: pd.NA})

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

print(df.head())