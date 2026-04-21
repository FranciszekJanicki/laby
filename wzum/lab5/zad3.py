import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic.csv")
df = df.drop(columns=['boat', 'body', 'home.dest'])

X = df.drop(columns=['survived'])
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)