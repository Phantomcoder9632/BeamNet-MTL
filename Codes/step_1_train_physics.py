import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# --- DYNAMIC PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# We are now loading the new PHYSICS dataset!
dataset_path = os.path.join(project_root, 'Dataset', 'physics_5g_users.csv')
train_file = os.path.join(project_root, 'Dataset', 'Step_1_train_data.csv')
test_file = os.path.join(project_root, 'Dataset', 'Step_1_test_data.csv')
models_dir = os.path.join(project_root, 'Models')
model_path = os.path.join(models_dir, 'step_1_physics_model.keras')

# --- CUDA VERIFICATION ---
print("===========================================")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ CUDA IS ACTIVE! Found GPU: {gpus[0].name}")
else:
    print("⚠️ CUDA not detected. Falling back to CPU.")
print("===========================================\n")

# --- 1. LOAD THE DATASET ---
print(f"📂 Loading physics dataset from {dataset_path}...")
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{dataset_path}'. Did you run step_1_physics_data.py?")
    exit()

# --- 2. SPLIT AND SAVE TRAIN/TEST DATA ---
print("✂️ Splitting data into Training (80%) and Testing (20%) sets...")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)
print(f"💾 Saved Step 1 training and testing data to Dataset folder.\n")

# --- 3. PREPARE DATA FOR NEURAL NETWORK ---
print("🧠 Preparing features (X, Y, RSRP) and labels (Target Beam)...")
# THE BIG UPGRADE: We now feed the AI 3 pieces of information per user!
X_train = train_df[['X_Location', 'Y_Location', 'RSRP_dBm']].values
y_train = train_df['Target_Beam'].values

X_test = test_df[['X_Location', 'Y_Location', 'RSRP_dBm']].values
y_test = test_df['Target_Beam'].values

NUM_BEAMS = 64

# --- 4. BUILD THE UPGRADED MODEL ---
print("⚙️ Building the Physics-Aware Neural Network...")
model = Sequential([
    # Notice the input_shape is now (3,) instead of (2,) because we added RSRP!
    Dense(128, activation='relu', input_shape=(3,)), 
    Dense(256, activation='relu'), # Made the brain slightly bigger to handle the complex physics
    Dense(NUM_BEAMS, activation='softmax') 
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# --- 5. TRAIN THE MODEL ---
print("🚀 Training the AI on local hardware...")
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

# --- 6. EVALUATE ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Final Test Accuracy on Noisy Physics Data: {accuracy * 100:.2f}%")

# --- 7. SAVE THE TRAINED MODEL ---
print("\n📦 Saving the trained model...")
os.makedirs(models_dir, exist_ok=True)
model.save(model_path)
print(f"✅ Model successfully saved to: {model_path}")