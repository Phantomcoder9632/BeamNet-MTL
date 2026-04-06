import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Simulate 1,000 users standing around a cell tower
NUM_USERS = 1000
NUM_BEAMS = 64

print("📡 Generating synthetic users...")
# Generate random X and Y coordinates (between -100 meters and 100 meters)
X_coords = np.random.uniform(-100, 100, NUM_USERS)
Y_coords = np.random.uniform(-100, 100, NUM_USERS)

# 2. Calculate the "Angle" of the user from the tower
# We use arctan2 to find the angle in radians (-pi to +pi)
angles = np.arctan2(Y_coords, X_coords)

# 3. Assign a Beam ID based on the angle
# We divide the full 360-degree circle into 64 slices (our 64 beams)
beam_bins = np.linspace(-np.pi, np.pi, NUM_BEAMS + 1)
beam_ids = np.digitize(angles, beam_bins) - 1 

# Keep Beam IDs strictly between 0 and 63
beam_ids = np.clip(beam_ids, 0, NUM_BEAMS - 1)

# 4. Save to a simple DataFrame
df = pd.DataFrame({
    'X_Location': X_coords,
    'Y_Location': Y_coords,
    'Target_Beam': beam_ids
})

# --- NEW ADDITIONS FOR LOCAL WORKFLOW ---

# 5. Save the dataset to a physical file in your folder
csv_filename = 'synthetic_5g_users.csv'
df.to_csv(csv_filename, index=False)
print(f"✅ Dataset saved locally as: {csv_filename}")
print(df.head())

# 6. VISUALIZATION: Prove the math worked!
print("\n📊 Generating visualization...")
plt.figure(figsize=(10, 8))

# Scatter plot: x, y, and color ('c') based on the Beam ID. 
# 'hsv' is a rainbow colormap that works perfectly for 360 degrees.
scatter = plt.scatter(df['X_Location'], df['Y_Location'], c=df['Target_Beam'], cmap='hsv', alpha=0.7)

# Mark the cell tower right in the center at (0,0)
plt.plot(0, 0, marker='^', color='black', markersize=15, label='5G Tower (gNB)')

plt.title(f"Top-Down View of {NUM_USERS} Users Colored by Assigned Beam (0-63)")
plt.xlabel("X Coordinate (Meters)")
plt.ylabel("Y Coordinate (Meters)")
plt.colorbar(scatter, label="Optimal Beam ID")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save the plot as an image in your folder, then open it on your screen
plt.savefig('beam_sectors_plot.png', dpi=300)
print("✅ Plot saved locally as: beam_sectors_plot.png")
plt.show()