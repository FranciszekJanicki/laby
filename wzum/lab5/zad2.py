import pandas as pd

df = pd.read_csv("titanic.csv")

df = df.drop(columns=['boat', 'body', 'home.dest'])

df = df.rename(columns={
    'pclass': 'ticket_class'
})

print(df.head())