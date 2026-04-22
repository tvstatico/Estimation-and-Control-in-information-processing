import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
np.random.seed(42)
N = 50
t = np.arange(N)

# --- State & Observation Generation ---
# Logic: x[k+1] = x[k] + 1 starting at 0 is identical to np.arange(N)
x = np.arange(N) 
noise = np.random.normal(loc=0, scale=2, size=N)
y = x + noise

# --- Visualization ---
plt.figure(figsize=(10, 6))

# Plotting the ground truth line
plt.plot(t, x, label="True State (x)", color='tab:blue', linewidth=2, zorder=2)

# Plotting the noisy measurements
plt.scatter(t, y, label="Noisy Observations (y)", color='tab:orange', 
            edgecolor='k', alpha=0.7, s=30, zorder=3)

plt.title("True State vs. Noisy Observations", fontsize=14)
plt.xlabel("Time step (k)", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()