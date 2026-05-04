from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from src.logistic import tune_logistic
from src.KNN import tune_knn
from src.RandomForest import tune_rf
from src.best_model import choose_best_model

def main():
    print("Loading data...")
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n=== Parameter Tuning ===")
    lr_model, lr_acc = tune_logistic(X_train, y_train, X_test, y_test)
    knn_model, knn_acc = tune_knn(X_train, y_train, X_test, y_test)
    rf_model, rf_acc = tune_rf(X_train, y_train, X_test, y_test)

    results = {
        "Logistic Regression": (lr_model, lr_acc),
        "KNN": (knn_model, knn_acc),
        "Random Forest": (rf_model, rf_acc)
    }

    best_model = choose_best_model(results)

    models = list(results.keys())
    accuracies = [lr_acc, knn_acc, rf_acc]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison of Tuned Models')
    plt.ylim(0, 1.05)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center')
    plt.show()

if __name__ == "__main__":
    main()