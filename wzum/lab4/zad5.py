from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt

X_iso = df[['plas', 'mass']].dropna()

iso = IsolationForest(contamination=0.05, random_state=42)
preds = iso.fit_predict(X_iso)

sns.scatterplot(x=X_iso['plas'], y=X_iso['mass'], hue=preds)
plt.title("Isolation Forest (1=inlier, -1=outlier)")
plt.show()