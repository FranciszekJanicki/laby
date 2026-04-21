import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("titanic.csv")
df = df.drop(columns=['boat', 'body', 'home.dest'])

X = df.drop(columns=['survived'])
y = df['survived']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

preds = np.random.randint(0, 2, size=len(y_test))

print("Accuracy:", accuracy_score(y_test, preds))