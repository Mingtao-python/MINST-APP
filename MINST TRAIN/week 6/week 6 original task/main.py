from src.data import load_data
from src.preprocess import standardize
from src.model import train_logistic
from src.evaluate import evaluate, show_wrong_predictions
from src.explain import explanation
import numpy as np

def main():
    X_train, X_test, y_train, y_test = load_data()

    model_before = train_logistic(X_train, y_train)
    acc_b, _, _ = evaluate(model_before, X_test, y_test)
    print("Accuracy BEFORE standardization:", acc_b)

    X_train_s, X_test_s = standardize(X_train, X_test)
    model_after = train_logistic(X_train_s, y_train)

    acc_a, report_str, y_pred_s = evaluate(model_after, X_test_s, y_test)
    print("Accuracy AFTER standardization:", acc_a)
    print(report_str)

    show_wrong_predictions(X_test, y_test, y_pred_s)
    wrong_idx = np.where(y_pred_s != y_test)[0]
    explanation(wrong_idx, y_test)
    print("Accuracy alone can hide class-level weaknesses.")

if __name__ == "__main__":
    main()