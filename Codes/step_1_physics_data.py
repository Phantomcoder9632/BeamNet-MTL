import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- DYNAMIC PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dataset_path = os.path.join(project_root, 'Dataset', 'physics_5g_users.csv')
plots_dir = os.path.join(project_root, 'Output_plots')

print("===========================================")
print("📡 GENERATING TELECOM PHYSICS DATASET")
print("===========================================\n")

NUM_USERS = 2000
NUM_BEAMS = 64

# 1. Place users (expanding the grid to 500 meters)
X_coords = np.random.uniform(-500, 500, NUM_USERS)
Y_coords = np.random.uniform(-500, 500, NUM_USERS)

# 2. Calculate true Distance and Angle
distances = np.sqrt(X_coords**2 + Y_coords**2)
angles = np.arctan2(Y_coords, X_coords)

# Assign Target Beams (Same as before)
beam_bins = np.linspace(-np.pi, np.pi, NUM_BEAMS + 1)
beam_ids = np.digitize(angles, beam_bins) - 1 
beam_ids = np.clip(beam_ids, 0, NUM_BEAMS - 1)

# --- THE TELECOM PHYSICS MATH ---
print("🧮 Calculating Log-Distance Path Loss and RSRP...")

# Standard 5G Sub-6GHz Parameters
P_tx = 23 # Transmit power of a mobile phone in dBm
L_0 = 38  # Signal lost in the first 1 meter (Path loss at reference distance)
n = 3.0   # Path Loss Exponent (3.0 represents an urban city environment)

# Add a little bit of random "Shadowing" noise (trees, cars, people moving)
shadowing_noise = np.random.normal(0, 4.0, NUM_USERS) # 4 dB standard deviation

# The Official Path Loss Formula
# PL = L_0 + 10 * n * log10(distance) + noise
path_loss = L_0 + 10 * n * np.log10(distances) + shadowing_noise

# Calculate Final RSRP (What the tower actually "hears")
# RSRP = Transmit Power - Path Loss
rsrp_dbm = P_tx - path_loss

# 3. Save to DataFrame
df = pd.DataFrame({
    'X_Location': X_coords,
    'Y_Location': Y_coords,
    'Distance_m': distances,
    'Target_Beam': beam_ids,
    'RSRP_dBm': rsrp_dbm
})

os.makedirs(os.path.join(project_root, 'Dataset'), exist_ok=True)
df.to_csv(dataset_path, index=False)
print(f"✅ Physics Dataset saved to: {dataset_path}")

# --- VISUALIZE THE PHYSICS ---
print("\n📊 Generating RSRP Heatmap...")
os.makedirs(plots_dir, exist_ok=True)
plt.figure(figsize=(10, 8))

# We color the dots based on Signal Strength (RSRP) now!
# 'viridis' goes from Dark Purple (Weak Signal) to Yellow (Strong Signal)
scatter = plt.scatter(df['X_Location'], df['Y_Location'], c=df['RSRP_dBm'], cmap='viridis', alpha=0.8)
plt.plot(0, 0, marker='^', color='red', markersize=15, label='5G Tower')

plt.title("5G User Signal Strength (RSRP in dBm) Decay Over Distance")
plt.xlabel("X Coordinate (Meters)")
plt.ylabel("Y Coordinate (Meters)")
# Add a colorbar so we can read the dBm values
cbar = plt.colorbar(scatter)
cbar.set_label("RSRP (dBm) -> Higher is Better")

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plot_path = os.path.join(plots_dir, 'step_1_rsrp_decay.png')
plt.savefig(plot_path, dpi=300)
print(f"✅ Plot saved to: {plot_path}")
plt.show()