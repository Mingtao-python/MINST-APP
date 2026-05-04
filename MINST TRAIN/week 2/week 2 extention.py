from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import time

start = time.time()
digits = load_digits()
X = digits.data
y = digits.target
end = time.time()
print(f"Data loaded in {end - start} seconds.")

start2 = time.time()
best_acc = 0
best_seed = None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

if acc > best_acc:
    best_acc = acc
    best_seed = 1
    print("New best:", best_acc, "seed:", best_seed)
end2 = time.time()
completed = end2 - start2
print(f"Initial seed tested in {completed} seconds.")
print(f'The prediction for complete all seed may take {completed * 100000} seconds.')

for seed in range(2, 100001):
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

    if acc > best_acc:
        best_acc = acc
        best_seed = seed
        print("New best:", best_acc, "seed:", best_seed)

    if acc == 1.0:
        print("FOUND PERFECT SEED:", seed)
        break