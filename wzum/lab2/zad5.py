import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("training_data.txt", header=None, names=["charged_time", "battery_lasted"])

X = df[["charged_time"]]
y = df["battery_lasted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color="blue", label="Rzeczywiste")
plt.scatter(X_test, y_pred, color="red", label="Predykcja")
plt.xlabel("Charged time")
plt.ylabel("Battery lasted")
plt.legend()
plt.title("Rzeczywiste vs Predykcja")
plt.show()