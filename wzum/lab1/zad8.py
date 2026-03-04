import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv("training_data.txt", header=None, names=["charged_time", "battery_lasted"])

print(df.head())

data = df[["charged_time"]]
target = df["battery_lasted"]

data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2)

print(data_train)
print(data_test)