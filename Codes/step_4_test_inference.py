import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# --- PATHING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
test_data_path = os.path.join(project_root, 'Dataset', 'real_physics_test.csv')
model_path = os.path.join(project_root, 'Models', 'step_4_real_physics_model.keras')
scaler_y_path = os.path.join(project_root, 'Dataset', 'scaler_y.pkl')

print("🧪 INITIALIZING ADVANCED RESNET PERFORMANCE TEST")

# 1. LOAD MODEL, DATA, AND SCALER
if not os.path.exists(model_path):
    print(f"❌ Error: Model not found at {model_path}. Train the ResNet model first!")
    exit()

model = tf.keras.models.load_model(model_path)
test_df = pd.read_csv(test_data_path)
scaler_y = joblib.load(scaler_y_path)

# Extract I/Q columns
signal_cols = [col for col in test_df.columns if col.startswith('I_') or col.startswith('Q_')]
X_test = test_df[signal_cols].values
loc_actual_scaled = test_df[['X_Location', 'Y_Location']].values
beam_actual = test_df['Target_Beam'].values

# 2. RUN INFERENCE
print(f"🏃 Running predictions on {len(X_test)} unseen users...")
preds = model.predict(X_test, batch_size=64)
loc_preds_scaled = preds[0]
# Get the Beam ID by finding the index of the highest probability
beam_preds = np.argmax(preds[1], axis=1)

# 3. CONVERT BACK TO PHYSICAL REALITY (METERS)
loc_actual_meters = scaler_y.inverse_transform(loc_actual_scaled)
loc_preds_meters = scaler_y.inverse_transform(loc_preds_scaled)

# 4. CALCULATE SCIENTIFIC METRICS
# Euclidean Distance: sqrt((x2-x1)^2 + (y2-y1)^2)
dist_errors = np.sqrt(np.sum((loc_actual_meters - loc_preds_meters)**2, axis=1))
mean_error = np.mean(dist_errors)
median_error = np.median(dist_errors)
beam_acc = np.mean(beam_preds == beam_actual) * 100

print("\n" + "="*40)
print("🏆  FINAL OPTIMIZED MODEL RESULTS  🏆")
print("="*40)
print(f"📡 Beam Selection Accuracy: {beam_acc:.2f}%")
print(f"📍 Mean Distance Error:    {mean_error:.4f} Meters")
print(f"📏 Median Distance Error:  {median_error:.4f} Meters")
print(f"🎯 Best Case Precision:    {np.min(dist_errors):.4f} Meters")
print(f"⚠️  Worst Case Error:       {np.max(dist_errors):.4f} Meters")
print("="*40)

# 5. VISUAL ERROR DISTRIBUTION (Histogram)
# This helps you see if most users are below 1m even if the mean is higher
plt.figure(figsize=(10, 5))
plt.hist(dist_errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_error:.2f}m')
plt.title("Distribution of Localization Errors")
plt.xlabel("Error in Meters")
plt.ylabel("Number of Users")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()