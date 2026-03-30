import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

df = pd.read_csv("training_data.txt", header=None, names=["charged_time", "battery_lasted"])

print(df.describe())

plt.scatter(df["charged_time"], df["battery_lasted"])
plt.xlabel("Charged time")
plt.ylabel("Battery lasted")
plt.show()

X = df[["charged_time"]]
y = df["battery_lasted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    "Linear": LinearRegression(),
    "Tree": DecisionTreeRegressor(),
    "Polynomial": make_pipeline(PolynomialFeatures(2), LinearRegression()),
    "RandomForest": RandomForestRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name, "R2:", r2_score(y_test, pred))