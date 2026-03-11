from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

y = [0, 0, 0, 1]   # AND

clf = DecisionTreeClassifier()
clf.fit(X, y)

plt.figure(figsize=(6,4))
plot_tree(clf,
          feature_names=["x1","x2"],
          class_names=["0","1"],
          filled=True)

plt.show()