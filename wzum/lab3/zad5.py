import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

iris = load_iris(as_frame=True)
df = iris.frame

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1])
plt.title("MinMaxScaler")
plt.show()