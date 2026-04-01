import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data
df = load_data()

cols_with_missing = ['plas', 'pres', 'skin', 'insu', 'mass']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

sns.histplot(df['mass'], bins=30)
plt.title("Histogram BMI (mass)")
plt.show()

sns.boxplot(x=df['mass'])
plt.title("Boxplot BMI (mass)")
plt.show()