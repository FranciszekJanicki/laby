from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt

digits = load_digits()

print(digits.DESCR)

print("shape of data:", digits.data.shape)
print("shape of images:", digits.images.shape)
print("target names:", digits.target_names)

print("\nfrom data (flattened image - 64 features):")
print(digits.data[0])

print("\nfrom images (8x8 matrix):")
print(digits.images[0])

plt.imshow(digits.images[0], cmap="gray")
plt.title("digit: " + str(digits.target[0]))
plt.show()

fig, axes = plt.subplots(2, 5, figsize=(10,5))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title("label: " + str(digits.target[i]))
    ax.axis("off")

plt.show()

X_train = digits.data[:5]
y_train = digits.target[:5]

clf = SVC()
clf.fit(X_train, y_train)

X_test = digits.data[6].reshape(1, -1)

prediction = clf.predict(X_test)

print("predicted:", prediction[0])
print("actual:", digits.target[6])

plt.imshow(digits.images[6], cmap="gray")
plt.title("prediction: " + str(prediction[0]))
plt.show()