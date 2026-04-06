import os
import numpy as np
import pandas as pd

# --- DYNAMIC PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dataset_path = os.path.join(project_root, 'Dataset', 'step_3_synthetic_iq_data.csv')

print("===========================================")
print("📡 GENERATING I/Q RADIO SIGNAL DATASET")
print("===========================================\n")

NUM_USERS = 5000
NUM_BEAMS = 64
NUM_ANTENNAS = 64

print(f"Generating synthetic radio waves for {NUM_USERS} users...")

# 1. Generate Physical Ground Truth (X, Y, and Target Beam)
X_coords = np.random.uniform(-500, 500, NUM_USERS)
Y_coords = np.random.uniform(-500, 500, NUM_USERS)

angles = np.arctan2(Y_coords, X_coords)
beam_bins = np.linspace(-np.pi, np.pi, NUM_BEAMS + 1)
target_beams = np.digitize(angles, beam_bins) - 1
target_beams = np.clip(target_beams, 0, NUM_BEAMS - 1)

# 2. Simulate the Complex Radio Signal (The H-Matrix proxy)
# We embed the X/Y locations into the wave math so the AI has a pattern to discover
print("✂️ Performing I/Q Split across 64 antennas...")
I_data = np.sin(X_coords[:, None] / 100 + np.random.randn(NUM_USERS, NUM_ANTENNAS))
Q_data = np.cos(Y_coords[:, None] / 100 + np.random.randn(NUM_USERS, NUM_ANTENNAS))

# 3. Package it all into a massive DataFrame
# We create names for all 128 antenna columns (I_0, I_1... Q_0, Q_1...)
I_cols = [f'I_{i}' for i in range(NUM_ANTENNAS)]
Q_cols = [f'Q_{i}' for i in range(NUM_ANTENNAS)]

df_targets = pd.DataFrame({
    'X_Location': X_coords, 
    'Y_Location': Y_coords, 
    'Target_Beam': target_beams
})
df_signals = pd.DataFrame(np.concatenate((I_data, Q_data), axis=1), columns=I_cols + Q_cols)

# Glue the targets and the 128 signal columns together side-by-side
df_final = pd.concat([df_targets, df_signals], axis=1)

# 4. Save to physical file
os.makedirs(os.path.join(project_root, 'Dataset'), exist_ok=True)
df_final.to_csv(dataset_path, index=False)
print(f"✅ Master I/Q Dataset saved to: {dataset_path}")
print(f"Dataset shape: {df_final.shape} (Users, Targets + 128 Signals)")