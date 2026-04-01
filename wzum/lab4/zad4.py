from scipy.stats import zscore
import numpy as np

df_z = df.copy()
df_z[['plas', 'mass']] = df_z[['plas', 'mass']].apply(zscore)

df_filtered = df[(np.abs(df_z[['plas', 'mass']]) < 3).all(axis=1)]

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=df_filtered['plas'], y=df_filtered['mass'])
plt.title("Po usuniêciu outlierów (z-score)")
plt.show()