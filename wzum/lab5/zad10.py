import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

sns.histplot(data=df, x='age', hue='survived', bins=30)
plt.show()

sns.boxplot(x='survived', y='fare', data=df)
plt.show()