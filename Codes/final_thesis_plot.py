import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# Load everything
model = tf.keras.models.load_model(r'D:\5G_project\Models\step_4_real_physics_model.keras')
test_df = pd.read_csv(r'D:\5G_project\Dataset\real_physics_test.csv')
scaler_y = joblib.load(r'D:\5G_project\Dataset\scaler_y.pkl')

# Get predictions
signal_cols = [col for col in test_df.columns if col.startswith('I_') or col.startswith('Q_')]
preds = model.predict(test_df[signal_cols].values)

actual_meters = scaler_y.inverse_transform(test_df[['X_Location', 'Y_Location']])
preds_meters = scaler_y.inverse_transform(preds[0])
errors = np.sqrt(np.sum((actual_meters - preds_meters)**2, axis=1))

# --- THE HERO PLOT ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: True vs Predicted
ax1.scatter(actual_meters[:, 0], actual_meters[:, 1], color='#2ecc71', alpha=0.4, s=5, label='Actual UE')
ax1.scatter(preds_meters[:, 0], preds_meters[:, 1], color='#e74c3c', alpha=0.3, s=5, label='AI Prediction')
ax1.set_title("Uplink Spatial Reconstruction", fontsize=14, fontweight='bold')
ax1.set_xlabel("Street X-Axis (m)")
ax1.set_ylabel("Street Y-Axis (m)")
ax1.legend(loc='upper right', markerscale=5)

# Plot 2: Heatmap
sc = ax2.scatter(actual_meters[:, 0], actual_meters[:, 1], c=errors, cmap='inferno', s=8)
cbar = plt.colorbar(sc, ax=ax2)
cbar.set_label('Euclidean Distance Error (Meters)', fontsize=12)
ax2.set_title("Geospatial Error Heatmap", fontsize=14, fontweight='bold')
ax2.set_xlabel("Street X-Axis (m)")
ax2.set_ylabel("Street Y-Axis (m)")

plt.tight_layout()
plt.savefig(r'D:\5G_project\Dataset\final_thesis_plot.png', dpi=300)
plt.show()