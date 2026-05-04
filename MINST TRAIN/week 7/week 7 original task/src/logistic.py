from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def tune_logistic(X_train, y_train, X_test, y_test):
    best_acc = 0
    best_model = None
    best_params = None

    for C in [0.1, 1, 5]:
        for max_iter in [1000, 3000, 5000]:
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = (C, max_iter)

    print(f"[Logistic Regression] Best params: C={best_params[0]}, max_iter={best_params[1]}")
    return best_model, best_acc