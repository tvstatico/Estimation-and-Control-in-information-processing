import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
N = 50
t = np.arange(N)

# --- State Generation ---
x = np.cumsum(np.ones(N)) - 1 

# --- Visualization ---
plt.figure("Task 1 - Evolution of the true state", figsize=(8, 5))
plt.plot(t, x, linewidth=2, color='tab:blue', label='True State')

plt.title("State Evolution Over Time")
plt.xlabel("Time stamp")
plt.ylabel("State")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()