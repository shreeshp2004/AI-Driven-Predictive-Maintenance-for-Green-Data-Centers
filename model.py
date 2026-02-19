# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
import joblib

# ✅ Load Dataset
file_path = "AEP_hourly.csv"  # Ensure this file is in the same directory
df = pd.read_csv(file_path)

# ✅ Convert Datetime column to actual datetime type
df["Datetime"] = pd.to_datetime(df["Datetime"])

# ✅ Extract time-based features
df["Year"] = df["Datetime"].dt.year
df["Month"] = df["Datetime"].dt.month
df["Day"] = df["Datetime"].dt.day
df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.weekday

# ✅ Drop unnecessary columns
df.drop(columns=["Datetime"], inplace=True)

# ✅ Check for missing values
df.dropna(inplace=True)

# ✅ Define Features & Target Variable
X = df.drop(columns=["AEP_MW"])  # Features (excluding target)
y = df["AEP_MW"]  # Target Variable

# ✅ Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# ✅ Best Model Training
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# ✅ Make Predictions
y_pred = best_model.predict(X_test)

# ✅ Evaluate Model Performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ✅ Classification Metrics: Convert Regression to Binary Classification
threshold = np.median(y)  # Define threshold based on median value
y_test_class = (y_test >= threshold).astype(int)  # High (1) or Low (0) Power Consumption
y_pred_class = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
conf_matrix = confusion_matrix(y_test_class, y_pred_class)

# ✅ Print Performance Metrics
print("\n🔹 Regression Metrics:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"✅ R-Squared (R² Score): {r2:.2%}")

print("\n🔹 Classification Metrics (Binary Categorization of Power Consumption):")
print(f"✅ Accuracy: {accuracy:.2%}")
print(f"✅ Precision: {precision:.2%}")
print(f"✅ Recall: {recall:.2%}")
print(f"✅ F1-Score: {f1:.2%}")
print("\nConfusion Matrix:")
print(conf_matrix)

# ✅ Save the best trained model
joblib.dump(best_model, "optimized_carbon_footprint_model.pkl")
print("\n✅ Optimized Model saved as 'optimized_carbon_footprint_model.pkl'")

# ✅ Generate and Display All Graphs

# 1️⃣ Energy Consumption Trends
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['AEP_MW'], color='blue', alpha=0.6)
plt.xlabel('Index')
plt.ylabel('Power Consumption (MW)')
plt.title('Energy Consumption Trends Over Time')
plt.grid()
plt.show()

# 2️⃣ Carbon Emission Trends (Example Data)
# Assuming carbon emissions are proportional to energy consumption
carbon_emissions = df['AEP_MW'] * 0.5  # Example: 0.5 tons of CO₂ per MW
plt.figure(figsize=(12, 5))
plt.plot(df.index, carbon_emissions, color='red', alpha=0.6)
plt.xlabel('Index')
plt.ylabel('Carbon Emissions (tons of CO₂)')
plt.title('Carbon Emission Trends Over Time')
plt.grid()
plt.show()

# 3️⃣ Confusion Matrix Visualization
plt.figure(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Low", "High"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Binary Classification")
plt.show()

# 4️⃣ Training vs Validation Accuracy and Loss (Example Data)
# Assuming you have training history from a neural network or other model
history = {
    'accuracy': [0.75, 0.85, 0.90, 0.92, 0.94],  # Training accuracy
    'val_accuracy': [0.70, 0.80, 0.85, 0.88, 0.90],  # Validation accuracy
    'loss': [0.5, 0.3, 0.2, 0.15, 0.1],  # Training loss
    'val_loss': [0.6, 0.4, 0.3, 0.25, 0.2]  # Validation loss
}

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

print("\n✅ All graphs generated successfully!")