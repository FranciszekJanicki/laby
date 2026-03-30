from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = load_iris(as_frame=True)
df = iris.frame

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC())
])

pipe.fit(X_train, y_train)

print("Accuracy:", pipe.score(X_test, y_test))