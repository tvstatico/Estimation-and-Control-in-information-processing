# =============================================================================
# Task 1. Build a simple linear dynamic system
# Objective: define the system model
# Students must write a small MATLAB or Python script for the system
# x(t+1)=Ax(t)+Bu(t)+w(t)
# y(t)=Cx(t)+v(t)
# Use a very simple one dimensional case first:
#     A=1
#     B=1
#     C=1
#     u(t)=1 for all time steps
#     initial state x(0)=0
# Add small random process noise w(t) and measurement noise v(t).
# 
# 1. Generate the true state for 30 time steps.
# 2. Generate the noisy measurements.
# 3. Plot true state and measured output on the same graph.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Reproducibility (remove or change seed to see different noise realizations)
np.random.seed(0)

# ---- System parameters ----
A = 1.0
B = 1.0
C = 1.0

N = 30          # number of time steps
u = 1.0         # constant input

# Noise standard deviations (small)
sigma_w = 0.1   # process noise std
sigma_v = 0.3   # measurement noise std

# ---- Pre-allocate arrays ----
x = np.zeros(N+1)   # true state, x[0]..x[N]
y = np.zeros(N+1)   # noisy measurement

# ---- Initial condition ----
x[0] = 0.0
v0 = sigma_v * np.random.randn()
y[0] = C * x[0] + v0

# ---- Simulate the system ----
for t in range(N):
    w = sigma_w * np.random.randn()   # process noise
    v = sigma_v * np.random.randn()   # measurement noise

    x[t+1] = A * x[t] + B * u + w     # state update
    y[t+1] = C * x[t+1] + v           # noisy measurement

# ---- Plot ----
time = np.arange(N+1)

plt.figure(figsize=(9, 5))
plt.plot(time, x, 'b-o', label='True state x(t)', linewidth=2, markersize=5)
plt.plot(time, y, 'r--x', label='Measured output y(t)', linewidth=1.5, markersize=7)
plt.xlabel('Time step t')
plt.ylabel('Value')
plt.title('Linear Dynamic System: True State vs. Noisy Measurement')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()