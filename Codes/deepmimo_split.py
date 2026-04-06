import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # Useful for saving the scalers for later use

# --- PATHING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
raw_dataset_path = os.path.join(project_root, 'Dataset', 'step_3_real_deepmimo_iq.csv')
output_dir = os.path.join(project_root, 'Dataset')

print("🛰️ Loading Raw DeepMIMO Dataset...")
df = pd.read_csv(raw_dataset_path)

# 1. SEPARATE FEATURES AND TARGETS
signal_cols = [col for col in df.columns if col.startswith('I_') or col.startswith('Q_')]
X = df[signal_cols].values
Y_loc = df[['X_Location', 'Y_Location']].values
Y_beam = df['Target_Beam'].values

# 2. APPLY SCALING (Standardization)
# We fit on ALL data here because this is our static research environment
scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
Y_loc_scaled = scaler_y.fit_transform(Y_loc)

# 3. THREE-WAY SPLIT (70/15/15)
# First split: Separate 15% for the final "Hidden Test" set
X_temp, X_test, Y_loc_temp, Y_loc_test, Y_beam_temp, Y_beam_test = train_test_split(
    X_scaled, Y_loc_scaled, Y_beam, test_size=0.15, random_state=42
)

# Second split: From the remaining 85%, take 17.6% for Validation (~15% of total)
X_train, X_val, Y_loc_train, Y_loc_val, Y_beam_train, Y_beam_val = train_test_split(
    X_temp, Y_loc_temp, Y_beam_temp, test_size=0.176, random_state=42
)

# 4. HELPER FUNCTION TO PACKAGE AND SAVE
def save_split(X_data, Loc_data, Beam_data, name):
    # Combine signals, locations, and beams back into one DataFrame
    df_signals = pd.DataFrame(X_data, columns=signal_cols)
    df_targets = pd.DataFrame({
        'X_Location': Loc_data[:, 0],
        'Y_Location': Loc_data[:, 1],
        'Target_Beam': Beam_data
    })
    df_final = pd.concat([df_targets, df_signals], axis=1)
    
    file_path = os.path.join(output_dir, f'real_physics_{name}.csv')
    df_final.to_csv(file_path, index=False)
    print(f"✅ Saved {name} set to: {file_path} ({len(df_final)} rows)")

# 5. EXECUTE SAVING
save_split(X_train, Y_loc_train, Y_beam_train, "train")
save_split(X_val, Y_loc_val, Y_beam_val, "val")
save_split(X_test, Y_loc_test, Y_beam_test, "test")

# Save the scalers (Essential for turning AI predictions back into meters later)
joblib.dump(scaler_x, os.path.join(output_dir, 'scaler_x.pkl'))
joblib.dump(scaler_y, os.path.join(output_dir, 'scaler_y.pkl'))

print("\n🚀 Pre-processing Complete. You now have scaled Train/Val/Test CSVs!")