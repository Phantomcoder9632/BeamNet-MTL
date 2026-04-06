import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- PATHING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
dataset_dir = os.path.join(project_root, 'Dataset')
model_save_path = os.path.join(project_root, 'Models', 'step_3_real_physics_model.keras')

def load_split(name):
    path = os.path.join(dataset_dir, f'real_physics_{name}.csv')
    df = pd.read_csv(path)
    signal_cols = [col for col in df.columns if col.startswith('I_') or col.startswith('Q_')]
    X = df[signal_cols].values
    Y_loc = df[['X_Location', 'Y_Location']].values
    Y_beam = df['Target_Beam'].values
    return X, Y_loc, Y_beam

print("🛰️ Loading Pre-Split Scaled Datasets...")
X_train, loc_train, beam_train = load_split("train")
X_val, loc_val, beam_val = load_split("val")

# --- 1. MODEL ARCHITECTURE ---
inputs = Input(shape=(128,), name='raw_radio_iq')
x = Dense(512, activation='elu')(inputs)
x = BatchNormalization()(x)
x = Dense(256, activation='elu')(x)
x = Dropout(0.3)(x) # Increased dropout to 30% to fight overfitting harder
shared = Dense(128, activation='elu')(x)

# Head 1: Localization
loc_x = Dense(128, activation='elu')(shared)
loc_x = Dense(64, activation='elu')(loc_x)
loc_out = Dense(2, name='location_output')(loc_x)

# Head 2: Beam Prediction
beam_x = Dense(128, activation='elu')(shared)
beam_out = Dense(64, activation='softmax', name='beam_output')(beam_x)

model = Model(inputs=inputs, outputs=[loc_out, beam_out])

# --- 2. COMPILE ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={'location_output': 'mse', 'beam_output': 'sparse_categorical_crossentropy'},
    metrics={'location_output': 'mae', 'beam_output': 'accuracy'}
)

# --- 3. THE "ANTI-OVERFIT" CALLBACKS ---
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10,          # Stop if no improvement after 10 epochs
    restore_best_weights=True, # Roll back to the best version of the model
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,           # Cut learning rate by 80% if stuck
    patience=5,           # Wait 5 epochs before cutting
    min_lr=1e-6,          # Don't go lower than this
    verbose=1
)

# --- 4. TRAIN ---
print(f"🚀 Training with Overfitting Protection on {len(X_train)} samples...")
model.fit(
    X_train, {'location_output': loc_train, 'beam_output': beam_train},
    validation_data=(X_val, {'location_output': loc_val, 'beam_output': beam_val}),
    epochs=150,           # Set higher, EarlyStopping will handle the cut
    batch_size=32,
    callbacks=[early_stop, reduce_lr], # NEW FIX
    verbose=1
)

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"✅ Training Complete. Model saved to: {model_save_path}")