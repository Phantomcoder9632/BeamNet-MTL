import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split

# --- DYNAMIC PATHING SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Safely pointing to your synthetic dataset
dataset_path = os.path.join(project_root, 'Dataset', 'step_3_synthetic_iq_data.csv')
models_dir = os.path.join(project_root, 'Models')
model_path = os.path.join(models_dir, 'step_3_synthetic_multitask_model.keras')

print("===========================================")
print("🧠 TRAINING MULTI-TASK NEURAL NETWORK")
print("===========================================\n")

# --- 1. LOAD THE DATASET ---
print(f"📂 Loading I/Q dataset from {dataset_path}...")
try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{dataset_path}'.")
    exit()

# --- 2. EXTRACT INPUTS AND MULTIPLE TARGETS ---
print("✂️ Separating inputs (128 raw signals) from targets (Location & Beam)...")

signal_cols = [col for col in df.columns if col.startswith('I_') or col.startswith('Q_')]
X_signals = df[signal_cols].values

locations = df[['X_Location', 'Y_Location']].values
target_beams = df['Target_Beam'].values

X_train, X_test, loc_train, loc_test, beam_train, beam_test = train_test_split(
    X_signals, locations, target_beams, test_size=0.2, random_state=42
)

NUM_BEAMS = 64

# --- 3. BUILD THE MULTI-TASK MODEL (FUNCTIONAL API) ---
print("⚙️ Assembling the Multi-Task Architecture...")

inputs = Input(shape=(128,), name='raw_radio_iq')

x = Dense(256, activation='relu')(inputs)
x = Dense(128, activation='relu')(x)
shared_features = Dense(128, activation='relu')(x)

loc_output = Dense(2, name='location_output')(shared_features)
beam_output = Dense(NUM_BEAMS, activation='softmax', name='beam_output')(shared_features)

model = Model(inputs=inputs, outputs=[loc_output, beam_output])

# --- 4. COMPILE AND TRAIN ---
model.compile(
    optimizer='adam',
    loss={
        'location_output': 'mse', 
        'beam_output': 'sparse_categorical_crossentropy' 
    },
    metrics={
        'location_output': 'mae', 
        'beam_output': 'accuracy'
    }
)

print("🚀 Training both tasks simultaneously...")
history = model.fit(
    X_train, 
    {'location_output': loc_train, 'beam_output': beam_train},
    validation_data=(X_test, {'location_output': loc_test, 'beam_output': beam_test}),
    epochs=50,
    verbose=1
)

# --- 5. EVALUATE & SAVE (THE FIX IS HERE) ---
print("\n📊 Final Evaluation on Test Data:")
# THE BUG FIX: return_dict=True maps the metrics safely by their variable names!
results = model.evaluate(X_test, {'location_output': loc_test, 'beam_output': beam_test}, verbose=0, return_dict=True)

print(f"🎯 Total Combined Loss: {results['loss']:.4f}")
print(f"📍 Location MAE (Meters off target): {results['location_output_mae']:.2f}m")
print(f"📡 Beam Prediction Accuracy: {results['beam_output_accuracy'] * 100:.2f}%")

print("\n📦 Saving the Multi-Task model...")
os.makedirs(models_dir, exist_ok=True)
model.save(model_path)
print(f"✅ Model successfully saved to: {model_path}")