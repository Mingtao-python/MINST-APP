from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

best_acc = 0
best_seed = None
best_cm = None
print('Testing seeds...')

for seed in range(0, 501):
    if seed % 100 == 0:
        print(f"Testing seed: {seed}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    if acc > best_acc:
        best_acc = acc
        best_seed = seed
        best_cm = cm

print("Best accuracy:", best_acc)
print("Best random_state:", best_seed)
print("Best confusion matrix:")
print(best_cm)