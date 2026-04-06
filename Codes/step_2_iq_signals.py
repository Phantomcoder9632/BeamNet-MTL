import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("===========================================")
print("📡 GENERATING RAW I/Q RADIO SIGNALS")
print("===========================================\n")

NUM_USERS = 5 # Let's just look at 5 users to keep it simple
NUM_ANTENNAS = 64

print(f"Simulating {NUM_USERS} users sending a signal to {NUM_ANTENNAS} antennas...\n")

# 1. Simulate the Complex Radio Wave (The H-Matrix)
# In real life, this comes from the DeepMIMO dataset. 
# Here, we generate random complex numbers: (Real + Imaginary * j)
real_part = np.random.randn(NUM_USERS, NUM_ANTENNAS)
imaginary_part = np.random.randn(NUM_USERS, NUM_ANTENNAS)

# This is the raw, complex radio signal hitting the tower!
H_matrix = real_part + 1j * imaginary_part 

print("🛑 WHAT THE TOWER HEARS (COMPLEX NUMBERS):")
print(f"User 1, Antenna 1 Signal: {H_matrix[0, 0]:.2f}")
print("AI Cannot read this! It will crash.\n")

# 2. Digital Signal Processing: The I/Q Split
print("✂️ Performing I/Q Split (Separating Real and Imaginary)...")
I_data = np.real(H_matrix) # Extract In-Phase
Q_data = np.imag(H_matrix) # Extract Quadrature

# 3. Flattening for the Neural Network
# We glue the 64 'I' values and 64 'Q' values together into a single list of 128 numbers
ai_input_data = np.concatenate((I_data, Q_data), axis=1)

print("✅ WHAT THE AI WILL SEE (REAL NUMBERS):")
print(f"User 1's flattened feature vector size: {ai_input_data[0].shape}")
print(f"First 5 numbers: {ai_input_data[0, :5]}")
print("The AI can now process this perfectly!\n")

# --- VISUALIZATION: Plotting the Radio Wave ---
print("📊 Plotting the I/Q Signal for User 1...")
plt.figure(figsize=(12, 5))

# Plot the 64 In-Phase signals
plt.plot(I_data[0], label='In-Phase (I) - Real', marker='o', color='blue', alpha=0.7)
# Plot the 64 Quadrature signals
plt.plot(Q_data[0], label='Quadrature (Q) - Imaginary', marker='x', color='red', alpha=0.7)

plt.title("Raw Radio Signal Across 64 Antennas for a Single User")
plt.xlabel("Antenna Index (0 to 63)")
plt.ylabel("Signal Amplitude")
plt.axhline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()