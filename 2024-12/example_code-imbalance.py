import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate a synthetic dataset with class imbalance
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=2, n_redundant=10,
                           n_classes=2,
                           weights=[0.95, 0.05],  # 95% of class 0, 5% of class 1
                           flip_y=0, random_state=42)
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution:")
print(class_distribution)

#--- CASE 1: High precision low recall model

# Simple split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 2. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 3. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision_hp = precision_score(y_test, y_pred)
recall_hp = recall_score(y_test, y_pred)
f1_hp = f1_score(y_test, y_pred)
arithmetic_mean_hp = (precision_hp + recall_hp) / 2

print(f"Logistic Regression Model:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision_hp:.2f}")
print(f"Recall: {recall_hp:.2f}")
print(f"F1 Score (Harmonic Mean): {f1_hp:.2f}")
print(f"Arithmetic Mean of Precision and Recall: {arithmetic_mean_hp:.2f}")
