# save_encoder.py
import pandas as pd
import joblib

# 1. Load your dataset
df = pd.read_csv("Churn.csv")

# 2. Apply the same preprocessing as during model training
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))

# 3. Save the encoder (basically the column names after one-hot encoding)
encoder_info = {"columns": X.columns.tolist()}
joblib.dump(encoder_info, "encoder.pkl")

print("Encoder saved as encoder.pkl")
