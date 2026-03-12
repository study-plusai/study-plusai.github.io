# Model training and dataset preprocessing script
# Run this to prepare the models for the backend

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import math

# Load dataset
import os
DATA_PATH = os.path.join(os.path.dirname(__file__), 'StudentPerformanceFactors.csv')
df = pd.read_csv(DATA_PATH)


# Clean and preprocess
# Remove missing/invalid values for selected columns
cols = ['Hours_Studied', 'Previous_Scores', 'Motivation_Level', 'Sleep_Hours', 'Exam_Score']
df = df.dropna(subset=cols)
df = df[cols]

# Encode Motivation_Level (Low/Medium/High) to numeric
df['Motivation_Level'] = df['Motivation_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})

# Input features and target
X = df[['Hours_Studied', 'Previous_Scores', 'Motivation_Level', 'Sleep_Hours']]
y = df['Exam_Score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression MAE: {mae:.2f}")
print(f"Linear Regression RMSE: {rmse:.2f}")
print(f"Linear Regression R2: {r2:.2f}")

# Save model
joblib.dump(lr, 'lr_model.joblib')

# Decision Tree Model (priority classification)
# For prototype, create priority labels based on study hours
priority_labels = pd.cut(y, bins=[0, 2, 5, np.inf], labels=['Critical', 'Optimize', 'Stable'])
df['priority'] = priority_labels

X_cls = X
y_cls = df['priority']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
dt = DecisionTreeClassifier()
dt.fit(X_train_cls, y_train_cls)

# Evaluate
y_pred_cls = dt.predict(X_test_cls)
accuracy = (y_pred_cls == y_test_cls).mean()
print(f"Decision Tree Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(dt, 'dt_model.joblib')
