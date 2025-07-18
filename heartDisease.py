import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv('heart.csv')
data = data.drop_duplicates()

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# Save model and scaler
joblib.dump(model, 'model_joblib_heart')
joblib.dump(scaler, 'scaler_joblib_heart')
joblib.dump(X.columns.tolist(), 'features_joblib_heart')
