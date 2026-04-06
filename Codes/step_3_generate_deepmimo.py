import os
import numpy as np
import pandas as pd
import deepmimo as DeepMIMO

# --- DYNAMIC PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Pointing exactly to your 'Dataset' folder
dataset_folder_path = os.path.join(project_root, 'Dataset')
output_csv_path = os.path.join(dataset_folder_path, 'step_3_real_deepmimo_iq.csv')

print("===========================================")
print("📡 GENERATING REAL DEEPMIMO RAY-TRACING DATA")
print("===========================================\n")

# --- 1. LOAD SCENARIO AND TRIM DATASET ---
print("⚙️ Configuring DeepMIMO O1_60 Scenario...")
try:
    # CONFIG FIX: Point DeepMIMO to your master 'Dataset' folder
    DeepMIMO.config.set('scenarios_folder', dataset_folder_path)
    # Selective load BS 1 (ID 5) and merge urban grids RX_1, RX_2, RX_3
    # This solves inhomogeneous shape errors and saves 20x RAM
    dataset = DeepMIMO.load('o1_60', tx_sets=[5])
    dataset = dataset.merge()
except Exception as e:
    print(f"\n❌ Scenario loading failed: {e}")
    print("Trying to fetch o1_60 automatically...")
    dataset = DeepMIMO.load('o1_60')
    dataset = dataset.merge()

# CRITICAL FIX for v4: Select users BEFORE computing channels to avoid RAM crashes
print("✂️ Selecting user subset (Rows 1-500)...")
# Mode "row" selects specific rows in the scenario grid
idxs = dataset.get_idxs("row", row_idxs=range(1, 501))
dataset = dataset.trim(idxs=idxs)

parameters = DeepMIMO.ChannelParameters()

# Set Base Station (We only have one BS loaded now, so index is 0)
parameters.active_BS = np.array([0])

# Antennas: 64 at the Base Station, 1 at the User equipment
# Using the v4 attribute access for antenna shape
parameters.bs_antenna['shape'] = np.array([1, 64, 1]) 
parameters.ue_antenna['shape'] = np.array([1, 1, 1])

# Enable 1 OFDM subcarrier in v4
# Using 1024 subcarriers increases the symbol duration (FFT window) 
# to ensure the path delays are captured without being clipped to zero.
parameters.ofdm.subcarriers = 1024
parameters.ofdm.selected_subcarriers = np.array([0])
parameters.ofdm.bandwidth = 0.5e9 # 500 MHz

# --- 2. GENERATE THE CHANNEL DATA ---
print("🚀 Running Ray-Tracing Channel Generation...")
try:
    # In v4, we compute channels on the TRIMMED dataset
    ch_data = dataset.compute_channels(parameters)
except Exception as e:
    print(f"\n❌ Error generating DeepMIMO: {e}")
    exit()

# --- 3. EXTRACT TARGETS (LOCATIONS) AND SIGNALS (H-MATRIX) ---
print("✂️ Extracting H-matrices and performing I/Q split...")

# In v4, attributes like ue_pos and channels are directly available
# ue_pos shape is (n_ue, 3), channels shape is (n_ue, n_rx, n_tx, n_freqs)
X_coords = dataset.ue_pos[:, 0]
Y_coords = dataset.ue_pos[:, 1]

# Reshape channels to (n_ue, 64) since n_rx=1 and n_freqs=1
# Flattening the last three dimensions for each user
H_matrices = dataset.channels.reshape(len(X_coords), -1)

# Do the I/Q Split!
I_data = np.real(H_matrices)
Q_data = np.imag(H_matrices)

NUM_USERS = len(X_coords)
print(f"✅ Successfully processed {NUM_USERS} valid users from the ray-tracing grid.")

# --- 4. CALCULATE BEAM IDs (GROUND TRUTH) ---
NUM_BEAMS = 64
# Calculate angles using the Base Station as the origin (0,0 in O1 scenario)
angles = np.arctan2(Y_coords, X_coords)
beam_bins = np.linspace(-np.pi, np.pi, NUM_BEAMS + 1)
target_beams = np.digitize(angles, beam_bins) - 1
target_beams = np.clip(target_beams, 0, NUM_BEAMS - 1)

# --- 5. PACKAGE INTO CSV ---
print("📦 Packaging into CSV format...")
I_cols = [f'I_{i}' for i in range(I_data.shape[1])]
Q_cols = [f'Q_{i}' for i in range(Q_data.shape[1])]

df_targets = pd.DataFrame({
    'X_Location': X_coords, 
    'Y_Location': Y_coords, 
    'Target_Beam': target_beams
})
df_signals = pd.DataFrame(np.concatenate((I_data, Q_data), axis=1), columns=I_cols + Q_cols)

df_final = pd.concat([df_targets, df_signals], axis=1)

df_final.to_csv(output_csv_path, index=False)
print(f"🎉 MASTER DEEPMIMO DATASET SAVED: {output_csv_path}")