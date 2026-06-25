#visuals

import os
import csv  
import numpy as np
import matplotlib
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split

OUT_DIR = "visuals"
os.makedirs(OUT_DIR, exist_ok=True)  # ensure output directory exists

FEATURE_NAME = [
    "nose_x", "nose_y",
    "lshoulder_x", "lshoulder_y",
    "rshoulder_x", "rshoulder_y",
    "lelbow_x", "lelbow_y",
    "relbow_x", "relbow_y",
    "lwrist_x", "lwrist_y",
    "rwrist_x", "rwrist_y",
]

X, y = [], [] #load data (the rows has a label)

with open("data/gesture_data.csv", newline = "") as csvfile:
    for row in csv.reader(csvfile):
        if not row:
            continue
        X.append([float(x) for x in row[1:]]) # features
        y.append(row[0]) # label

X, y = np.array(X), np.array(y)
labels = sorted(set(y))
model = joblib.load("models/gesture_model.joblib")

# feature importance diagram (histogram)
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
order = np.argsort(importances)[::-1]
plt.bar([FEATURE_NAME[i] for i in order], importances[order])
plt.ylabel("Feature Importance")
plt.title("Feature Importance in Gesture Classification")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"))
plt.close()

# random forest diagram
plt.figure(figsize=(60,30))
plot_tree(model.estimators_[0], feature_names=FEATURE_NAME, class_names=labels, filled=True, rounded=True, fontsize=10, impurity=False)
plt.title("Decision Tree from Random Forest")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/decision_tree.svg")
plt.savefig(f"{OUT_DIR}/decision_tree.png", dpi=100)
plt.close()

plt.figure(figsize=(30,20))
plot_tree(model.estimators_[0],max_depth=3, feature_names=FEATURE_NAME, class_names=labels, filled=True, rounded=True, fontsize=10, impurity=False)
plt.title("Decision Tree from Random Forest (Depth=3)")
plt.savefig(os.path.join(OUT_DIR, "decision_tree_depth3.png"))
plt.tight_layout()

plt.close()


print(f"Visualizations saved in {OUT_DIR} directory.")