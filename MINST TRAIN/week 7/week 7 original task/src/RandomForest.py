from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def tune_rf(X_train, y_train, X_test, y_test):
    best_acc = 0
    best_model = None
    best_params = None

    for n in [50, 100, 200]:
        for depth in [None, 10, 20]:
            model = RandomForestClassifier(n_estimators=n, max_depth=depth)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = (n, depth)

    print(f"[Random Forest] Best params: n_estimators={best_params[0]}, max_depth={best_params[1]}")
    return best_model, best_acc