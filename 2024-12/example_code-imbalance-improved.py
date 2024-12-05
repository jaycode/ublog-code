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

#--- CASE 2: Good prediction model
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 3. Calculate Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
arithmetic_mean = (precision + recall) / 2

print(f"Random Forest Model:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score (Harmonic Mean): {f1:.2f}")
print(f"Arithmetic Mean of Precision and Recall: {arithmetic_mean:.2f}")
print()



#--- CASE 3: Improved technique
import pandas as pd


# Combine features and target for easier manipulation
X = pd.DataFrame(X)
y = pd.Series(y)
df = pd.concat([X, y], axis=1)
df.columns = [f'feature_{i}' for i in range(df.shape[1] - 1)] + ['target']

# 1.a. Handling duplicates
print(f"Original data shape: {df.shape}")
df.drop_duplicates(inplace=True)
print(f"Data shape after removing duplicates: {df.shape}")

# 1.b. Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# 1.c. Feature Scaling

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Identify categorical columns (columns 1, 2, and 3 are categorical)
categorical_cols = ['feature_1', 'feature_2', 'feature_3']
numerical_cols = [col for col in df.columns[:-1] if col not in categorical_cols]

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature Scaling
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 2.a. Feature Selection
from sklearn.ensemble import RandomForestClassifier

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Encode the target variable
y = le.fit_transform(y)

# Train a Random Forest to get feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
print("Top 10 features based on importance:")
print(feature_importance_df.head(10))

# Select top 10 features
top_features = feature_importance_df['Feature'].head(10).tolist()
X = X[top_features]

# 2.b. Feature Extraction

from sklearn.preprocessing import PolynomialFeatures

# Create interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# 3. Algorithm Tuning

from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)

# Fit Grid Search
grid_search.fit(X_poly, y)

# Best parameters
print(f"Best parameters found: {grid_search.best_params_}")

# 4. Boosting

from sklearn.ensemble import GradientBoostingClassifier

# Initialize Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)

# Define parameter grid for Gradient Boosting
param_grid_gbc = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.05],
    'max_depth': [3, 5],
}

# Initialize Grid Search
grid_search_gbc = GridSearchCV(estimator=gbc, param_grid=param_grid_gbc, cv=3, scoring='f1', n_jobs=-1)

# Fit Grid Search
grid_search_gbc.fit(X_poly, y)

# Best parameters
print(f"Best parameters found for Gradient Boosting: {grid_search_gbc.best_params_}")

# 5. Regularization
# 5.a. L1 and L2 Regularization

from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression with L1 regularization
lr = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', random_state=42)

# Fit model
lr.fit(X_poly, y)

# 5.b. Dropout (in Neural Networks)
from sklearn.neural_network import MLPClassifier

# Initialize Neural Network with L2 regularization
mlp = MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, random_state=42, max_iter=300)

# Fit model
mlp.fit(X_poly, y)

# 6. Model Evaluation

from sklearn.metrics import classification_report

# Split data into training and test sets for evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42, stratify=y)

# Best Random Forest model
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Best Gradient Boosting model
best_gbc = grid_search_gbc.best_estimator_
best_gbc.fit(X_train, y_train)
y_pred_gbc = best_gbc.predict(X_test)
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, y_pred_gbc))

# Logistic Regression
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Neural Network
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_mlp))

# 7. Results and Improvements

# Calculate F1 scores for each model on the test set
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
f1_gbc = f1_score(y_test, y_pred_gbc, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

print("Model Performance (Weighted F1 Scores):")
print(f"Random Forest: {f1_rf:.4f}")
print(f"Gradient Boosting: {f1_gbc:.4f}")
print(f"Logistic Regression: {f1_lr:.4f}")
print(f"Neural Network: {f1_mlp:.4f}")