# Exercise 2. Prediction step
# Implement the state prediction equation:
# x(t|t-1)=Ax(t-1|t-1)+Bu(t)
# Use x(0|0)=0.
# Compute the predicted state at each time step, without any update step.
# Plot the true state and the predicted state on the same graph.

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
x_true = np.zeros(N+1)   # true state
y_meas = np.zeros(N+1)   # noisy measurements (not used here, kept for context)
x_pred = np.zeros(N+1)   # predicted state x(t|t-1)

# ---- Initial conditions ----
x_true[0] = 0.0
x_pred[0] = 0.0          # x(0|0) = 0
y_meas[0] = C * x_true[0] + sigma_v * np.random.randn()

# ---- Simulate true system AND run prediction ----
for t in range(N):
    # True system (with noise)
    w = sigma_w * np.random.randn()
    v = sigma_v * np.random.randn()
    x_true[t+1] = A * x_true[t] + B * u + w
    y_meas[t+1] = C * x_true[t+1] + v

    # Prediction step (no noise, no measurement correction)
    # x(t+1 | t) = A * x(t | t) + B * u
    # Without an update step, x(t|t) equals the previous prediction
    x_pred[t+1] = A * x_pred[t] + B * u

# ---- Plot ----
time = np.arange(N+1)

plt.figure(figsize=(9, 5))
plt.plot(time, x_true, 'b-o', label='True state x(t)', linewidth=2, markersize=5)
plt.plot(time, x_pred, 'g-s', label='Predicted state x(t|t-1)',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.xlabel('Time step t')
plt.ylabel('Value')
plt.title('Prediction-Only: True State vs. Predicted State')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Print the prediction error to see the drift ----
err = x_true - x_pred
print(f"Final prediction error at t={N}: {err[-1]:.4f}")
print(f"Mean abs. error: {np.mean(np.abs(err)):.4f}")