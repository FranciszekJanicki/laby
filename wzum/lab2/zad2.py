from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

wine = load_wine()

print(wine.DESCR)

print("shape of data:", wine.data.shape)
print("shape of target:", wine.target.shape)
print("classes:", wine.target_names)

print("\nfeatures:")
print(wine.data[:5])

print("\nlabels:")
print(wine.target[:5])

X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.2,
    random_state=42
)

print("\ntraining samples:", X_train.shape)
print("testing samples:", X_test.shape)

model_svm = SVC()
model_svm.fit(X_train, y_train)

svm_score = model_svm.score(X_test, y_test)
print("\nSVM accuracy:", svm_score)

model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)

tree_score = model_tree.score(X_test, y_test)
print("decision Tree accuracy:", tree_score)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

knn_score = model_knn.score(X_test, y_test)
print("KNN accuracy:", knn_score)

prediction = model_svm.predict([X_test[0]])

print("\nprediction example:")
print("predicted:", prediction)
print("actual:", y_test[0])