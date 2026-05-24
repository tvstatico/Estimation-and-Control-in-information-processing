# Exercise 4. Full Kalman filter
# Implement the complete Kalman filter.
# Prediction:
#     x_hat(t|t-1)=Ax_hat(t-1|t-1)+Bu(t)
#     P(t|t-1)=AP(t-1|t-1)A^t+Q
# Update:
#     K(t)=P(t|t-1)C^T(CP(t|t-1)C^t+R)^(-1)
#     x_hat(t|t)=x_hat(t|t-1)+K(t)(y(t)-Cx_hat(t|t-1))
#     P(t|t)=(I-K(t)C)P(t|t-1)
# Use:
#     Q=0.01
#     R=0.5
#     P(0|0)=1
# Plot on the same figure:
#     true state
#     noisy measurement
#     Kalman estimate
# Also plot:
#     Kalman gain K(t)
#     covariance P(t|t)

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- System parameters ----
A = 1.0
B = 1.0
C = 1.0

N = 30
u = 1.0

# Filter-assumed noise covariances
Q = 0.01     # process noise variance
R = 0.5      # measurement noise variance

# True noise std devs used in the simulation (match Q and R)
sigma_w = np.sqrt(Q)       # 0.1
sigma_v = np.sqrt(R)       # ~0.707

# ---- Allocate arrays ----
x_true = np.zeros(N+1)     # true state
y_meas = np.zeros(N+1)     # noisy measurements
x_est  = np.zeros(N+1)     # x_hat(t|t) -- posterior estimate
P_est  = np.zeros(N+1)     # P(t|t)
K_gain = np.zeros(N+1)     # Kalman gain K(t)

# ---- Initial conditions ----
x_true[0] = 0.0
x_est[0]  = 0.0            # x_hat(0|0) = 0
P_est[0]  = 1.0            # P(0|0) = 1

y_meas[0] = C * x_true[0] + sigma_v * np.random.randn()
K_gain[0] = 0.0            # no update performed at t=0

# ---- Simulation + Kalman filter loop ----
for t in range(N):
    # --- True system (ground truth, unknown to the filter) ---
    w = sigma_w * np.random.randn()
    v = sigma_v * np.random.randn()
    x_true[t+1] = A * x_true[t] + B * u + w
    y_meas[t+1] = C * x_true[t+1] + v

    # --- Prediction step ---
    x_pred = A * x_est[t] + B * u
    P_pred = A * P_est[t] * A + Q

    # --- Update step ---
    S = C * P_pred * C + R                 # innovation covariance
    K = P_pred * C / S                     # Kalman gain
    innov = y_meas[t+1] - C * x_pred       # innovation
    x_est[t+1] = x_pred + K * innov
    P_est[t+1] = (1 - K * C) * P_pred

    K_gain[t+1] = K

time = np.arange(N+1)

# ======================================================================
# Plot 1: true state, measurements, Kalman estimate
# ======================================================================
plt.figure(figsize=(9, 5))
plt.plot(time, x_true, 'b-o',  label='True state x(t)',
         linewidth=2, markersize=5)
plt.plot(time, y_meas, 'rx',   label='Measurement y(t)',
         markersize=8, markeredgewidth=1.5)
plt.plot(time, x_est,  'g-s',  label='Kalman estimate x̂(t|t)',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.xlabel('Time step t')
plt.ylabel('Value')
plt.title('Kalman Filter: True State, Measurements, and Estimate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================================
# Plot 2: Kalman gain K(t) and covariance P(t|t)
# ======================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax1.plot(time[1:], K_gain[1:], 'b-o', linewidth=2, markersize=5)
ax1.set_ylabel('Kalman gain K(t)')
ax1.set_title('Kalman Gain and Posterior Covariance over Time')
ax1.grid(True)

ax2.plot(time, P_est, 'r-o', linewidth=2, markersize=5)
ax2.set_xlabel('Time step t')
ax2.set_ylabel('Covariance P(t|t)')
ax2.grid(True)

plt.tight_layout()
plt.show()

# ---- Some diagnostics ----
rmse_kalman = np.sqrt(np.mean((x_true - x_est)**2))
rmse_meas   = np.sqrt(np.mean((x_true - y_meas)**2))
print(f"RMSE of raw measurements vs. true state : {rmse_meas:.4f}")
print(f"RMSE of Kalman estimate vs. true state  : {rmse_kalman:.4f}")
print(f"Steady-state Kalman gain K(∞) ≈ {K_gain[-1]:.4f}")
print(f"Steady-state covariance P(∞) ≈ {P_est[-1]:.4f}")