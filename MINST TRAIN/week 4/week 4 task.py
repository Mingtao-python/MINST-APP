from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target
def PCA_():
  global X_train, X_test, y_train, y_test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  pca = PCA(n_components=2)
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
PCA_()

def LogisticRegression_():
  global model
  model = LogisticRegression(max_iter=5000)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'Accuracy: {accuracy}')

LogisticRegression_()

def Visualization():
  plt.figure(figsize=(15, 10))
  scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7)
  plt.colorbar(scatter, label='Digit Label (0-9)')
  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.title('PCA 2D Visualization of Digits Dataset (Training Data)')
  plt.grid(True)
  plt.show()

Visualization()

print('''
============ Explanation ===========
1. Why different digits remain separated after PCA:
  PCA finds the most important information in the data (image), so it can still be separated successfully with the important features.
2. Why some points overlap on the plot:
  Some digits are visually similar, so their points appear close together or overlap.
  For example, 1 and 7 can both be fairly straight, and 3 and 8 both have curved shapes, making them harder to separate using only two principal components.
3. What is its use in real life:
  PCA is a very fast digit recognition method, it may be used for quick digit recognition tasks, such as in postal code recognition, where speed is more important than perfect accuracy.
4. Why it cannot be visualized easily before reducing the dimensionality?
  Humans cannot easily look and understand a dimension more than 3, and the original data had 64 dimensions.
''')