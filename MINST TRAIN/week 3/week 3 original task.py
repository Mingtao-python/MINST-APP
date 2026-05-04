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
#-------------------Logistic Regression-------------------
start1 = time.time()
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
sc1 = accuracy_score(y_test, pred)
print("Accuracy of Logistic Regression:", sc1)
end1 = time.time()
print(f"Logistic Regression tested in {end1 - start1} seconds.")
print(confusion_matrix(y_test, pred))
#---------------------------KNN---------------------------
start2 = time.time()
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)
sc2 = accuracy_score(y_test, pred)
print("Accuracy of KNN:", sc2)
end2 = time.time()
print(f"KNN tested in {end2 - start2} seconds.")
print(confusion_matrix(y_test, pred))
#-----------------------Random Forest----------------------
start3 = time.time()
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
sc3 = accuracy_score(y_test, pred)
print("Accuracy of Random Forest:", sc3)
end3 = time.time()
print(f"Random Forest tested in {end3 - start3} seconds.")
print(confusion_matrix(y_test, pred))
#-----------------------Decision Tree(extension)----------------------
start4 = time.time()
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
sc4 = accuracy_score(y_test, pred)
print("Accuracy of Decision Tree:", sc4)
end4 = time.time()
print(f"Decision Tree tested in {end4 - start4} seconds.")
print(confusion_matrix(y_test, pred))
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


print('Random Forest has the highest accuracy among the four models, because it is an conclusion of many different desition, while KNN is a very similar model (voting), so there accuracy is very close to RandomForest.')