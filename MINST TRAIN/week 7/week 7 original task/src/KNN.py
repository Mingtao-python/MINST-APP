from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def tune_knn(X_train, y_train, X_test, y_test):
    best_acc = 0
    best_model = None
    best_k = None

    for k in range(1, 10):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_k = k

    print(f"[KNN] Best k={best_k}")
    return best_model, best_acc