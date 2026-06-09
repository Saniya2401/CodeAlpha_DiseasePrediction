# Disease Prediction Project
# CodeAlpha Internship

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load Dataset
df = pd.read_csv("heart.csv")

# Display First 5 Rows
print("First 5 Rows:\n")
print(df.head())

# Dataset Information
print("\nDataset Info:\n")
print(df.info())

# Check Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# Example Prediction
sample_data = np.array([[52,1,2,172,199,1,1,162,0,0.5,2,0,2]])

sample_data = scaler.transform(sample_data)

prediction = model.predict(sample_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Person has Heart Disease")
else:
    print("Person does NOT have Heart Disease")