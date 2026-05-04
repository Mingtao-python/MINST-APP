import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sb

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred)
    worse = 1.0
    worse_key = None
    
    for key in map(str, range(10)):
        s_key = report_dict[key]
        if s_key['f1-score'] < worse:
            worse = s_key['f1-score']
            worse_key = key
        elif s_key['f1-score'] == worse and worse_key is not None:
            worse_key += f" and {key}"

    print(f"The worse performing class is {worse_key} with an F1-score of {worse:.4f}.")
    return acc, report_str, y_pred

def show_wrong_predictions(X_test, y_test, y_pred):
    wrong_idx = np.where(y_pred != y_test)[0]

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(wrong_idx[:8]):
        plt.subplot(2, 4, i+1)
        plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
        plt.title(f"True:{y_test[idx]}, Pred:{y_pred[idx]}")
        plt.axis('off')
    plt.show()