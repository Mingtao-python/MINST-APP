import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report, y_pred

def show_wrong_predictions(X_test, y_test, y_pred):
    wrong_idx = np.where(y_pred != y_test)[0]

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(wrong_idx[:8]):
        plt.subplot(2, 4, i+1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        plt.title(f"True:{y_test[idx]}, Pred:{y_pred[idx]}")
        plt.axis('off')
    plt.show()