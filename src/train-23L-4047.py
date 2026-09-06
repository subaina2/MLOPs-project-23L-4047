import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Student ID Verification
STUDENT_ID = "23L-4047"
print(f"--- Starting Training Pipeline for Student ID: {STUDENT_ID} ---")

# 1. Load Data from data/ directory
data_path = os.path.join("data", "dataset.csv")
print(f"Loading raw dataset from {data_path}...")
df = pd.read_csv(data_path)

X = df[['feature1', 'feature2']] / 100 # Scaling Technique B (23L-4047)


X = df[['feature1', 'feature2']]
y = df['target']

# 2. Train ML Model
print("Training RandomForest model...")
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# 3. Serialize and Save Trained Model into model/ directory
os.makedirs("model", exist_ok=True)
model_output_path = os.path.join("model", f"model_{STUDENT_ID}.pkl")
joblib.dump(model, model_output_path)

print(f"Model successfully trained and saved to: {model_output_path}")
