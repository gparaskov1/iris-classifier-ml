import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
#data = pd.read_csv('/data/iris.csv')
iris = load_iris()

data = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
data['species'] = iris.target
print(data)

# Features and labels
X = data.drop('species', axis=1)
y = data['species']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, '../iris_model.pkl')

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

os.makedirs("outputs", exist_ok=True)

# X = features, y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_true = y_test
y_pred = model.predict(X_test)

os.makedirs("outputs", exist_ok=True)
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.savefig("../outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

os.makedirs("outputs", exist_ok=True)

# model = ...  # your trained estimator/pipeline
joblib.dump(model, "../outputs/model.joblib")