import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score

from data_loader import load_data
df = load_data()

print(df.head())
print(df.describe())
print(df.info())

print("Braki danych:\n", df.isnull().sum())

cols_with_missing = ['plas', 'pres', 'skin', 'insu', 'mass']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "DecisionTree": DecisionTreeClassifier(),
    "SVM": SVC()
}

print("\n=== BASELINE (fillna=0) ===")
for name, model in models.items():
    model.fit(X_train.fillna(0), y_train)
    preds = model.predict(X_test.fillna(0))
    print(name, accuracy_score(y_test, preds))

X_train_drop = X_train.dropna()
y_train_drop = y_train[X_train_drop.index]

print("\n=== DROP NA ===")
for name, model in models.items():
    model.fit(X_train_drop, y_train_drop)
    preds = model.predict(X_test.fillna(0))
    print(name, accuracy_score(y_test, preds))

imp = SimpleImputer(strategy='mean')
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.transform(X_test)

print("\n=== SIMPLE IMPUTER ===")
for name, model in models.items():
    model.fit(X_train_imp, y_train)
    preds = model.predict(X_test_imp)
    print(name, accuracy_score(y_test, preds))

knn = KNNImputer(n_neighbors=5)
X_train_knn = knn.fit_transform(X_train)
X_test_knn = knn.transform(X_test)

print("\n=== KNN IMPUTER ===")
for name, model in models.items():
    model.fit(X_train_knn, y_train)
    preds = model.predict(X_test_knn)
    print(name, accuracy_score(y_test, preds))