from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt

start0 = time.time()
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
end0 = time.time()
print(f"Data loaded and split in {end0 - start0} seconds.")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ------------------- Logistic Regression -------------------
start1 = time.time()
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
sc1 = accuracy_score(y_test, lr_pred)
print("Accuracy of Logistic Regression:", sc1)
end1 = time.time()
print(f"Logistic Regression tested in {end1 - start1} seconds.")
print(confusion_matrix(y_test, lr_pred))

# --------------------------- KNN ---------------------------
start2 = time.time()
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
sc2 = accuracy_score(y_test, knn_pred)
print("Accuracy of KNN:", sc2)
end2 = time.time()
print(f"KNN tested in {end2 - start2} seconds.")
print(confusion_matrix(y_test, knn_pred))

# ----------------------- Random Forest ----------------------
start3 = time.time()
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
sc3 = accuracy_score(y_test, rf_pred)
print("Accuracy of Random Forest:", sc3)
end3 = time.time()
print(f"Random Forest tested in {end3 - start3} seconds.")
print(confusion_matrix(y_test, rf_pred))

# ----------------------- Decision Tree ----------------------
start4 = time.time()
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
sc4 = accuracy_score(y_test, dt_pred)
print("Accuracy of Decision Tree:", sc4)
end4 = time.time()
print(f"Decision Tree tested in {end4 - start4} seconds.")
print(confusion_matrix(y_test, dt_pred))

#-----------------------Accuracy Bar Chart----------------------
models = ['Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree']
accuracies = [sc1, sc2, sc3, sc4]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Models')
plt.ylim(0, 1.05)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, acc, ha='center')
plt.show()

print("""
Why do the models perform differently?

1. Random Forest performs the best because the digits dataset is high‑dimensional (64 features) and highly non‑linear. 
   Random Forest combines many decision trees, allowing it to learn complex curved boundaries while reducing overfitting. 
   This balance of flexibility and stability makes it ideal for image-like data.

2. KNN performs close to Random Forest because the digits dataset is small and well‑clustered. 
   KNN naturally handles non‑linear patterns by comparing each test sample to its nearest neighbors. 
   Since similar digits cluster together in pixel space, KNN works surprisingly well.

3. Logistic Regression performs worse because it is a linear model. 
   Handwritten digits require non‑linear decision boundaries, and many digits overlap in linear space 
   (for example, 3 vs 5 or 8 vs 9). 
   Therefore, Logistic Regression cannot fully separate the classes.

4. Decision Tree performs the worst because a single tree easily overfits on 64‑dimensional pixel data. 
   It memorizes training samples instead of learning general patterns, leading to unstable predictions 
   and lower test accuracy.""")