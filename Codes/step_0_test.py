import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- DYNAMIC PATHING SETUP ---
# Safely navigate your specific folder structure
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Define exact file paths
test_file = os.path.join(project_root, 'Dataset', 'Step0_test_data.csv')
model_path = os.path.join(project_root, 'Models', 'step_0_baseline_model.keras')
plots_dir = os.path.join(project_root, 'Output_plots')

# --- 1. LOAD THE SAVED DATA AND MODEL ---
print("===========================================")
print("📡 INITIALIZING AI INFERENCE PIPELINE")
print("===========================================\n")

print(f"📂 Loading test dataset: {test_file}")
try:
    test_df = pd.read_csv(test_file)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{test_file}'.")
    exit()

print(f"🧠 Loading trained AI model: {model_path}")
try:
    model = load_model(model_path)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 2. PREPARE THE INPUTS ---
# The AI only gets to see the X and Y coordinates!
X_test = test_df[['X_Location', 'Y_Location']].values
# We keep the real answers hidden to check the AI's work later
y_actual = test_df['Target_Beam'].values

# --- 3. RUN PREDICTIONS ---
print("\n🚀 Asking AI to predict Beam IDs...")
# The AI outputs 64 probabilities per user. 
raw_predictions = model.predict(X_test)

# We use np.argmax to find the Beam ID with the highest probability (e.g., "I am 99% sure it's Beam 42")
y_pred = np.argmax(raw_predictions, axis=1)

# --- 4. EVALUATE ACCURACY ---
# Calculate how many times the AI's prediction perfectly matched the ground truth
accuracy = np.mean(y_pred == y_actual)
print(f"\n🎯 Independent Test Accuracy: {accuracy * 100:.2f}%")

# --- 5. VISUAL PROOF (THE AHA! MOMENT) ---
print("\n📊 Generating Side-by-Side Comparison Plot...")
# Create the Output_plots folder if it doesn't exist yet
os.makedirs(plots_dir, exist_ok=True)

plt.figure(figsize=(14, 6))

# Subplot 1: What the real world looks like (Ground Truth)
plt.subplot(1, 2, 1)
plt.scatter(test_df['X_Location'], test_df['Y_Location'], c=y_actual, cmap='hsv', alpha=0.8)
plt.plot(0, 0, marker='^', color='black', markersize=15, label='5G Tower')
plt.title("Actual Target Beams (Ground Truth)")
plt.xlabel("X Coordinate (Meters)")
plt.ylabel("Y Coordinate (Meters)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Subplot 2: What the AI *thinks* the world looks like (Predictions)
plt.subplot(1, 2, 2)
plt.scatter(test_df['X_Location'], test_df['Y_Location'], c=y_pred, cmap='hsv', alpha=0.8)
plt.plot(0, 0, marker='^', color='black', markersize=15, label='5G Tower')
plt.title(f"AI Predicted Beams (Accuracy: {accuracy * 100:.1f}%)")
plt.xlabel("X Coordinate (Meters)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Save and show the plot
plot_path = os.path.join(plots_dir, 'step_0_predictions_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✅ Visual comparison saved to: {plot_path}")
plt.show()