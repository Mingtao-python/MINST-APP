from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model_before_standardization():
  model = LogisticRegression(max_iter=2000)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  acc_before = accuracy_score(y_test, y_pred)
  print("Accuracy BEFORE standardization:", acc_before)
  cm_before = confusion_matrix(y_test, y_pred)
  print("Confusion Matrix BEFORE standardization:")
  print(cm_before)

def model_after_standardization():
  global y_pred_s
  scaler = StandardScaler()
  X_train_s = scaler.fit_transform(X_train)
  X_test_s = scaler.transform(X_test)

  model = LogisticRegression(max_iter=2000)
  model.fit(X_train_s, y_train)
  y_pred_s = model.predict(X_test_s)

  acc_after = accuracy_score(y_test, y_pred_s)
  print("Accuracy AFTER standardization:", acc_after)
  cm_after = confusion_matrix(y_test, y_pred_s)
  print("Confusion Matrix AFTER standardization:")
  print('There are not many differences between the standardized and non-standardized confusion matrix Because the pixel values in the digits dataset are already in a similar range so the model is not very sensitive to them')
  print(cm_after)
  print(classification_report(y_test, y_pred_s))

def wrong_predictions_show():
  wrong_idx = np.where(y_pred_s != y_test)[0]

  plt.figure(figsize=(10, 4))
  for i, idx in enumerate(wrong_idx[:8]):
      plt.subplot(2, 4, i+1)
      plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
      plt.title(f"True:{y_test[idx]}, Pred:{y_pred_s[idx]}")
      plt.axis('off')
  plt.show()

def explanation():
  print("""
  Based on my confusion matrix:

- The model often confuses 8 and 9.
  Reason: both have a closed loop and similar shape.

- It also confuses 3 and 5.
  Reason: some handwritten 3s look like a curved 5.

- 1 and 7 are occasionally mixed.
  Reason: some handwritten 1s have a serif at the top.

These mistakes match the visual similarity of handwritten digits.
""")

model_before_standardization()
model_after_standardization()
wrong_predictions_show()
explanation()
print("Some classes may still perform poorly even when the overall accuracy looks high, so accuracy alone can hide these problems.")