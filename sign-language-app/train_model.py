import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

data_dir = "data"
all_data = []
all_labels = []

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        label = file.split(".")[0]
        df = pd.read_csv(os.path.join(data_dir, file))
        all_data.append(df.values)
        all_labels.extend([label] * len(df))

X = np.vstack(all_data)
y = np.array(all_labels)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

os.makedirs("models", exist_ok=True)
with open("models/gesture_clf.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Training complete! Classifier saved as models/gesture_clf.pkl")
