# Exercise 3. Innovation
# Compute the innovation: y(t)-Cx(t|t-1).
# Calculate this quantity at each time step and plot it as a function of time.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- System parameters ----
A = 1.0
B = 1.0
C = 1.0

N = 30
u = 1.0

sigma_w = 0.1
sigma_v = 0.3

# ---- Allocate arrays ----
x_true     = np.zeros(N+1)
y_meas     = np.zeros(N+1)
x_pred     = np.zeros(N+1)   # x(t|t-1)
innovation = np.zeros(N+1)   # y(t) - C*x(t|t-1)

# ---- Initial conditions ----
x_true[0] = 0.0
x_pred[0] = 0.0
y_meas[0] = C * x_true[0] + sigma_v * np.random.randn()
innovation[0] = y_meas[0] - C * x_pred[0]

# ---- Simulate and compute innovation ----
for t in range(N):
    # True system
    w = sigma_w * np.random.randn()
    v = sigma_v * np.random.randn()
    x_true[t+1] = A * x_true[t] + B * u + w
    y_meas[t+1] = C * x_true[t+1] + v

    # Prediction step (no update -> x(t|t) = x(t|t-1))
    x_pred[t+1] = A * x_pred[t] + B * u

    # Innovation
    innovation[t+1] = y_meas[t+1] - C * x_pred[t+1]

# ---- Plot innovation vs. time ----
time = np.arange(N+1)

plt.figure(figsize=(9, 5))
plt.plot(time, innovation, 'm-o', linewidth=2, markersize=5,
         label='Innovation: y(t) - C x(t|t-1)')
plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Time step t')
plt.ylabel('Innovation')
plt.title('Innovation vs. Time (Prediction-Only Filter)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Diagnostics ----
print(f"Mean innovation:   {np.mean(innovation):.4f}")
print(f"Std. innovation:   {np.std(innovation):.4f}")
print(f"Final innovation:  {innovation[-1]:.4f}")