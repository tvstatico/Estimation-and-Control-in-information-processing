import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
np.random.seed(42)
N = 50
W = 5  # Window size
t = np.arange(N)

# --- State & Observation Generation ---
x = np.arange(N)  # Equivalent to the x[k+1] = x[k] + 1 loop
noise = np.random.normal(0, 2, N)
y = x + noise

# --- Moving Average Estimation ---
x_hat = np.array([np.mean(y[max(0, k - W + 1) : k + 1]) for k in range(N)])

# --- Visualization ---
plt.figure(figsize=(10, 6))

plt.plot(t, x, label="True State $x(k)$", color='black', linestyle='--', alpha=0.6)
plt.scatter(t, y, label="Noisy Observations $y(k)$", color='tab:red', s=20, alpha=0.5)
plt.plot(t, x_hat, label=f"Moving Average $\hat{{x}}$ (W={W})", color='tab:blue', linewidth=2.5)

plt.title("Smoothing Noisy Data with a Moving Average Filter", fontsize=14)
plt.xlabel("Time step $k$")
plt.ylabel("Value")
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()