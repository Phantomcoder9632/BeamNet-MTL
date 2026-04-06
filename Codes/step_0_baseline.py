import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# --- CUDA VERIFICATION ---
print("===========================================")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ CUDA IS ACTIVE! Found GPU: {gpus[0].name}")
else:
    print("⚠️ CUDA not detected. Falling back to CPU.")
print("===========================================\n")

# --- 1. LOAD THE DATASET ---
# Pointing to the file inside the Dataset folder
dataset_path = 'Dataset/synthetic_5g_users.csv'
print(f"📂 Loading dataset from {dataset_path}...")

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{dataset_path}'. Make sure your terminal is running from the root '5G_project' folder.")
    exit()

# --- 2. SPLIT AND SAVE TRAIN/TEST DATA ---
print("✂️ Splitting data into Training (80%) and Testing (20%) sets...")

# Split the DataFrame first so we can save it with the column headers intact
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the splits physically to the Dataset folder
train_file = 'Dataset/Step0_train_data.csv'
test_file = 'Dataset/Step0_test_data.csv'

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)
print(f"💾 Saved training data to: {train_file}")
print(f"💾 Saved testing data to: {test_file}\n")

# --- 3. PREPARE DATA FOR NEURAL NETWORK ---
print("🧠 Preparing features (X, Y) and labels (Target Beam) for the AI...")
# Extract the numbers from the dataframes into raw Numpy arrays for TensorFlow
X_train = train_df[['X_Location', 'Y_Location']].values
y_train = train_df['Target_Beam'].values

X_test = test_df[['X_Location', 'Y_Location']].values
y_test = test_df['Target_Beam'].values

NUM_BEAMS = 64 # We know there are 64 possible beams in our setup

# --- 4. BUILD THE MODEL ---
print("⚙️ Building the Neural Network...")
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)), 
    Dense(128, activation='relu'),
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
print(f"\n🎯 Final Test Accuracy: {accuracy * 100:.2f}%")

# --- 7. SAVE THE TRAINED MODEL ---
print("\n📦 Saving the trained model...")
# Create the Models folder if it doesn't already exist
os.makedirs('Models', exist_ok=True)

# Save the model in the recommended native Keras format
model_path = 'Models/step_0_baseline_model.keras'
model.save(model_path)
print(f"✅ Model successfully saved to: {model_path}")