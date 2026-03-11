# Step 1: Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Step 2: Load the Iris Dataset
iris = load_iris()
X = iris.data        # Features
y = iris.target      # Target labels

# Step 3: Split Dataset into Training and Testing Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Decision Tree Classifier Model
model = DecisionTreeClassifier(random_state=42)

# Step 5: Train the Model
model.fit(X_train, y_train)

# Step 6: Predict on Test Data
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Evaluation")
print("Accuracy:", round(accuracy, 4))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

# Step 8: Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Decision Tree")
plt.show()

