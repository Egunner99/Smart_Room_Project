import csv 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load(path):
    X, y = [], []
    with open(path, newline = "") as csvfile:
        for row in csv.reader(csvfile):
            if not row:
                continue
            X.append([float(x) for x in row[1:]]) # features
            y.append(row[0]) # label
    return np.array(X), np.array(y)

X_train, y_train = load("gesture_data.csv")
X_test, y_test = load("gesture_test.csv")
print (f"loaded {len(X_train)} training samples with {X_train.shape[1]} features each")

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

labels = sorted(set(y_test))
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))
print("Confusion matrix:")
cm = confusion_matrix(y_test, predictions, labels=labels)
print("Labels:", labels)
print(cm)
