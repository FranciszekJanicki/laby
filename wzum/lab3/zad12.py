from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris(as_frame=True)
df = iris.frame

feature_sets = {
    "sepal": ["sepal length (cm)", "sepal width (cm)"],
    "petal": ["petal length (cm)", "petal width (cm)"],
    "all": df.drop("target", axis=1).columns
}

for name, cols in feature_sets.items():
    X = df[cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    print(name, model.score(X_test, y_test))