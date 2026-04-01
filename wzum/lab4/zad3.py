import seaborn as sns
import matplotlib.pyplot as plt

from data_loader import load_data
df = load_data()

sns.scatterplot(x=df['plas'], y=df['mass'], hue=df['class'])
plt.title("mass vs plas")
plt.show()