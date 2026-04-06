import os
import time
import numpy as np
import tensorflow as tf

# --- PATHING ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
model_path = os.path.join(project_root, 'Models', 'step_4_real_physics_model.keras')
tflite_path = os.path.join(project_root, 'Models', 'optimized_model.tflite')

print("⚙️ Loading the heavy Python model...")
model = tf.keras.models.load_model(model_path)

# --- 1. CONVERT TO C++ OPTIMIZED TFLITE ---
print("🗜️ Compressing model for Edge Deployment (TFLite)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This removes training nodes and optimizes the math for CPU
tflite_model = converter.convert()

with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print("✅ TFLite model saved!")

# --- 2. LOAD THE OPTIMIZED MODEL ---
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

# Get input and output tracking locations
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a single "dummy" 5G user signal (1 user, 128 antennas)
dummy_signal = np.random.rand(1, 128).astype(np.float32)

# --- 3. THE WARM-UP RUN ---
# We run it once just to wake up the CPU cache
interpreter.set_tensor(input_details[0]['index'], dummy_signal)
interpreter.invoke()

# --- 4. THE TRUE LATENCY TEST ---
print("⏱️ Running true inference speed test...")

# We will run it 100 times and take the average to be perfectly scientific
latency_list = []

for _ in range(100):
    start_time = time.perf_counter()
    
    interpreter.set_tensor(input_details[0]['index'], dummy_signal)
    interpreter.invoke()
    # Outputs are here if we needed them: interpreter.get_tensor(output_details[0]['index'])
    
    end_time = time.perf_counter()
    latency_list.append((end_time - start_time) * 1000) # Convert to milliseconds

average_latency = np.mean(latency_list)
fastest_latency = np.min(latency_list)

print("\n" + "="*40)
print("⚡ PRODUCTION LATENCY PROOF ⚡")
print("="*40)
print(f"Average Inference Time: {average_latency:.4f} ms")
print(f"Fastest Inference Time: {fastest_latency:.4f} ms")
print("="*40)