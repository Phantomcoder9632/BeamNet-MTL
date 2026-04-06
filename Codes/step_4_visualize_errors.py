import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Load Model and Data
model = tf.keras.models.load_model(r'd:\5G_project\Models\step_4_real_physics_model.keras')
test_df = pd.read_csv(r'D:\5G_project\Dataset\real_physics_test.csv')
scaler_y = joblib.load(r'D:\5G_project\Dataset\scaler_y.pkl')

# Predict
signal_cols = [col for col in test_df.columns if col.startswith('I_') or col.startswith('Q_')]
preds = model.predict(test_df[signal_cols].values)

# Transform to Meters
actual_meters = scaler_y.inverse_transform(test_df[['X_Location', 'Y_Location']])
preds_meters = scaler_y.inverse_transform(preds[0])
errors = np.sqrt(np.sum((actual_meters - preds_meters)**2, axis=1))

# --- CREATE THE RESEARCH PLOT ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Prediction Accuracy (Spatial)
ax1.scatter(actual_meters[:, 0], actual_meters[:, 1], c='blue', alpha=0.3, s=2, label='True User Positions')
ax1.scatter(preds_meters[:, 0], preds_meters[:, 1], c='red', alpha=0.3, s=2, label='AI Predictions')
ax1.set_title("Spatial Reconstruction: AI 'Seeing' the Street")
ax1.set_xlabel("X (Meters)")
ax1.set_ylabel("Y (Meters)")
ax1.legend()

# Plot 2: The Heatmap (Performance Geography)
sc = ax2.scatter(actual_meters[:, 0], actual_meters[:, 1], c=errors, cmap='viridis', s=5)
plt.colorbar(sc, ax=ax2, label='Localization Error (Meters)')
ax2.set_title("Error Heatmap: Identifying 5G Shadow Zones")
ax2.set_xlabel("X (Meters)")
ax2.set_ylabel("Y (Meters)")

plt.tight_layout()
plt.savefig(r'D:\5G_project\Dataset\final_performance_map.png', dpi=300)
plt.show()

print("🎉 Final Hero Image saved to Dataset folder!")