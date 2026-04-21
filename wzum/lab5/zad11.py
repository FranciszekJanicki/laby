import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

df = df.replace({None: pd.NA})
df = df.drop(columns=['boat', 'body', 'home.dest', 'cabin'])
df = df.select_dtypes(include=['number'])

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()