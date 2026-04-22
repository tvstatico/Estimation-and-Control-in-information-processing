import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 1000

# --- Simulation ---
true_x = 10
noise = np.random.normal(0, 2, N) # Mean 0, std 2
y = true_x + noise

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.hist(y, bins=30, color='tab:purple', edgecolor='black', alpha=0.7)
plt.axvline(true_x, color='red', linestyle='--', linewidth=2, label=f'True State x={true_x}')

plt.title("Ex7: Distribution of Noisy Measurements")
plt.xlabel("Measured Value (y)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
