from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris(as_frame=True)
df = iris.frame

X = df[["sepal length (cm)", "sepal width (cm)"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model = SVC()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))