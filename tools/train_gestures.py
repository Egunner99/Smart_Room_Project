import csv
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

os .makedirs("models", exist_ok=True)  # ensure models directory exists

DATA_FILE = "data/gesture_data.csv"
MODEL_FILE = "models/gesture_model.joblib" # saving model here


X, y = [], [] #load data (the rows has a label)

with open(DATA_FILE, newline = "") as csvfile:
    for row in csv.reader(csvfile):
        if not row:
            continue
        X.append([float(x) for x in row[1:]]) # features
        y.append(row[0]) # label

X = np.array(X)
y = np.array(y)

print (f"loaded {len(X)} samples with {X.shape[1]} features each")

for label in sorted(set(y)):
    print(f"{label}: {sum(y == label)} samples")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# the straify=y ensures that the class distribution is preserved
# 80 train, 20 test, random state for reproducibility

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

cv = cross_val_score(clf, X, y, cv=5) # 5 fold cross validation
print(f"5-fold cross-val accuracy: {cv.mean():.1%} (+/- {cv.std():.1%})")

labels = sorted(set(y))
predictions = clf.predict(X_test)
print("Classification report:")
print(classification_report(y_test, predictions))
print("Confusion matrix:")
cm = confusion_matrix(y_test, predictions, labels=labels)
print("Labels:", labels)
print(cm)

joblib.dump(clf, "models/gesture_model_split.joblib")

final = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
final.fit(X, y) # train on all data
joblib.dump(final, "models/gesture_model.joblib")

print(f"Model saved to {MODEL_FILE}")

