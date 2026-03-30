import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

iris = load_iris(as_frame=True)
df = iris.frame

X = df[["sepal length (cm)", "sepal width (cm)"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "SVC": SVC(),
    "Tree": DecisionTreeClassifier(),
    "Forest": RandomForestClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    results[name] = acc

print(results)

with open("wyniki.json", "w") as f:
    json.dump(results, f)