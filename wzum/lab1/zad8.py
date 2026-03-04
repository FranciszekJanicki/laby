import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("training_data.txt", header=None, names=["charged_time", "battery_lasted"])

print(df.head())

data = df[["charged_time"]]
target = df["battery_lasted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)