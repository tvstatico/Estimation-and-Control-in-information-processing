# Exercise 5. Prediction only versus full Kalman filter
# Run the system in two cases:
#     1. prediction only
#     2. full Kalman filter
# For both cases, compute the mean squared error
# MSE = 1/N * sum_{t=1..N} (x(t)-x_hat(t))^2
# Compare the two results.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- System and filter parameters ----
A = 1.0
B = 1.0
C = 1.0

N = 30
u = 1.0

Q = 0.01          # filter's process-noise variance
R = 0.5           # filter's measurement-noise variance

sigma_w = np.sqrt(Q)
sigma_v = np.sqrt(R)

# ---- Allocate arrays ----
x_true = np.zeros(N+1)
y_meas = np.zeros(N+1)

x_pred_only = np.zeros(N+1)   # prediction-only estimate
x_kf        = np.zeros(N+1)   # full Kalman estimate
P           = np.zeros(N+1)   # Kalman posterior covariance
K_gain      = np.zeros(N+1)

# ---- Initial conditions (identical for both filters) ----
x_true[0]      = 0.0
x_pred_only[0] = 0.0
x_kf[0]        = 0.0
P[0]           = 1.0

y_meas[0] = C * x_true[0] + sigma_v * np.random.randn()

# ---- Simulate and run BOTH filters on the same data ----
for t in range(N):
    # --- True system ---
    w = sigma_w * np.random.randn()
    v = sigma_v * np.random.randn()
    x_true[t+1] = A * x_true[t] + B * u + w
    y_meas[t+1] = C * x_true[t+1] + v

    # --- Filter 1: prediction only (ignores measurements) ---
    x_pred_only[t+1] = A * x_pred_only[t] + B * u

    # --- Filter 2: full Kalman filter ---
    # Prediction
    x_pred = A * x_kf[t] + B * u
    P_pred = A * P[t] * A + Q
    # Update
    S = C * P_pred * C + R
    K = P_pred * C / S
    x_kf[t+1] = x_pred + K * (y_meas[t+1] - C * x_pred)
    P[t+1]    = (1 - K * C) * P_pred
    K_gain[t+1] = K

# ---- Compute MSE over t = 1..N ----
err_pred = x_true[1:] - x_pred_only[1:]
err_kf   = x_true[1:] - x_kf[1:]

MSE_pred = np.mean(err_pred**2)
MSE_kf   = np.mean(err_kf**2)

print("===== Mean Squared Error comparison =====")
print(f"Prediction-only  MSE : {MSE_pred:.4f}")
print(f"Full Kalman      MSE : {MSE_kf:.4f}")
print(f"Improvement factor   : {MSE_pred / MSE_kf:.2f}x")

# ---- Plot: both estimates vs. truth ----
time = np.arange(N+1)

plt.figure(figsize=(9, 5))
plt.plot(time, x_true,      'b-o', label='True state',
         linewidth=2, markersize=5)
plt.plot(time, y_meas,      'rx',  label='Measurement',
         markersize=7, markeredgewidth=1.2, alpha=0.6)
plt.plot(time, x_pred_only, 'm-s', label=f'Prediction only  (MSE={MSE_pred:.3f})',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.plot(time, x_kf,        'g-d', label=f'Kalman filter     (MSE={MSE_kf:.3f})',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.xlabel('Time step t')
plt.ylabel('Value')
plt.title('Prediction-only vs. Full Kalman Filter')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plot: squared errors over time ----
plt.figure(figsize=(9, 4))
plt.plot(time[1:], err_pred**2, 'm-s', label='Prediction-only squared error',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.plot(time[1:], err_kf**2,   'g-d', label='Kalman squared error',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.xlabel('Time step t')
plt.ylabel(r'$(x(t)-\hat{x}(t))^2$')
plt.title('Squared Error over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()