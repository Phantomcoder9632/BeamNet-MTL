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
model_path = os.path.join(project_root, 'Models', 'step_3_real_physics_model.keras')
scaler_y_path = os.path.join(project_root, 'Dataset', 'scaler_y.pkl')

print("🧪 INITIALIZING FINAL PERFORMANCE TEST")

# 1. LOAD MODEL, DATA, AND SCALER
model = tf.keras.models.load_model(model_path)
test_df = pd.read_csv(test_data_path)
scaler_y = joblib.load(scaler_y_path)

signal_cols = [col for col in test_df.columns if col.startswith('I_') or col.startswith('Q_')]
X_test = test_df[signal_cols].values
loc_actual_scaled = test_df[['X_Location', 'Y_Location']].values
beam_actual = test_df['Target_Beam'].values

# 2. RUN INFERENCE
print(f"🏃 Running predictions on {len(X_test)} unseen users...")
preds = model.predict(X_test)
loc_preds_scaled = preds[0]
beam_preds = np.argmax(preds[1], axis=1)

# 3. CALCULATE REAL-WORLD METRICS
# Inverse Transform the locations to get back to METERS
loc_actual_meters = scaler_y.inverse_transform(loc_actual_scaled)
loc_preds_meters = scaler_y.inverse_transform(loc_preds_scaled)

# Euclidean Distance Error (The "Ground Truth" Distance off target)
dist_errors = np.sqrt(np.sum((loc_actual_meters - loc_preds_meters)**2, axis=1))
avg_dist_error = np.mean(dist_errors)
beam_acc = np.mean(beam_preds == beam_actual) * 100

print("\n" + "!"*30)
print("🏆 FINAL UNBIASED PERFORMANCE 🏆")
print("!"*30)
print(f"📡 Beam Selection Accuracy: {beam_acc:.2f}%")
print(f"📍 Mean Distance Error: {avg_dist_error:.4f} Meters")
print(f"🎯 Best Case Error: {np.min(dist_errors):.4f} Meters")
print(f"⚠️ Worst Case Error: {np.max(dist_errors):.4f} Meters")
print("!"*30)

# 4. OPTIONAL: VISUALIZE PREDICTION SPREAD
plt.figure(figsize=(10, 6))
plt.scatter(loc_actual_meters[:, 0], loc_actual_meters[:, 1], color='blue', alpha=0.5, label='Actual Location')
plt.scatter(loc_preds_meters[:, 0], loc_preds_meters[:, 1], color='red', marker='x', alpha=0.5, label='AI Prediction')
plt.title("Joint Localization: Unseen Test Set Results")
plt.xlabel("X Coordinate (Meters)")
plt.ylabel("Y Coordinate (Meters)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()