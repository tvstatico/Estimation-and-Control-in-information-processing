# Exercise 9. Position and velocity estimation
# Use the two dimensional state 
# x(t)=[position
#       velocity]
# with
# A = [1 delta(t)
#      0 1]
# C = [1 0]
# Choose delta(t)=1.
# Simulate motion with constant velocity. Measure only the position, with noise.
# Use the Kalman filter to estimate both position and velocity.
# Plot:
#     true position and estimated position
#     true velocity and estimated velocity

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- Parameters ----
dt = 1.0
N  = 50                          # time steps

# State-space matrices
A = np.array([[1.0, dt],
              [0.0, 1.0]])
C = np.array([1.0, 0.0])         # 1D row: measure position only

# Noise covariances
# Small Q for the constant-velocity assumption -- we expect little
# process noise in an ideal constant-velocity regime.
Q = np.array([[0.001, 0.0],
              [0.0,   0.01]])    # a bit more uncertainty on velocity
R = 1.0                          # measurement variance (scalar)

# True initial state: at origin, moving at 1 unit per time step
x0_true = np.array([0.0, 1.0])

# ---- Simulate the true trajectory and noisy measurements ----
x_true = np.zeros((2, N+1))
y_meas = np.zeros(N+1)

x_true[:, 0] = x0_true
y_meas[0]    = C @ x_true[:, 0] + np.sqrt(R) * np.random.randn()

L = np.linalg.cholesky(Q)        # to generate correlated process noise
for t in range(N):
    w = L @ np.random.randn(2)
    x_true[:, t+1] = A @ x_true[:, t] + w
    y_meas[t+1]   = C @ x_true[:, t+1] + np.sqrt(R) * np.random.randn()

# ---- Kalman filter ----
x_est = np.zeros((2, N+1))
P     = np.zeros((2, 2, N+1))

# Initial estimate: we pretend we don't know the velocity
x_est[:, 0] = np.array([0.0, 0.0])         # wrong on velocity!
P[:, :, 0]  = np.diag([1.0, 10.0])         # large initial uncertainty on velocity

I2 = np.eye(2)

for t in range(N):
    # --- Prediction ---
    x_pred = A @ x_est[:, t]
    P_pred = A @ P[:, :, t] @ A.T + Q

    # --- Update ---
    innov = y_meas[t+1] - C @ x_pred       # scalar
    S     = C @ P_pred @ C + R             # scalar
    K     = P_pred @ C / S                 # shape (2,)

    x_est[:, t+1] = x_pred + K * innov
    P[:, :, t+1]  = (I2 - np.outer(K, C)) @ P_pred

# ---- Extract components for plotting ----
pos_true = x_true[0, :]
vel_true = x_true[1, :]
pos_est  = x_est[0, :]
vel_est  = x_est[1, :]

# 1-sigma uncertainty bands from the filter
pos_std = np.sqrt(P[0, 0, :])
vel_std = np.sqrt(P[1, 1, :])

time = np.arange(N+1)

# ======================================================================
# Plot: position and velocity on stacked axes
# ======================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# --- Position ---
ax1.plot(time, pos_true, 'b-',  label='True position',      linewidth=2)
ax1.plot(time, y_meas,   'rx',  label='Measured position',  markersize=7, alpha=0.6)
ax1.plot(time, pos_est,  'g-',  label='Estimated position', linewidth=2)
ax1.fill_between(time, pos_est - pos_std, pos_est + pos_std,
                 color='g', alpha=0.15, label=r'Estimate $\pm 1\sigma$')
ax1.set_ylabel('Position')
ax1.set_title('Position: True, Measured, and Kalman-Estimated')
ax1.legend(loc='upper left')
ax1.grid(True)

# --- Velocity ---
ax2.plot(time, vel_true, 'b-', label='True velocity',      linewidth=2)
ax2.plot(time, vel_est,  'g-', label='Estimated velocity', linewidth=2)
ax2.fill_between(time, vel_est - vel_std, vel_est + vel_std,
                 color='g', alpha=0.15, label=r'Estimate $\pm 1\sigma$')
ax2.axhline(1.0, color='b', linestyle=':', alpha=0.4)
ax2.set_xlabel('Time step t')
ax2.set_ylabel('Velocity')
ax2.set_title('Velocity: True and Kalman-Estimated (velocity is NOT measured!)')
ax2.legend(loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.show()

# ---- Diagnostics ----
rmse_pos = np.sqrt(np.mean((pos_true[1:] - pos_est[1:])**2))
rmse_vel = np.sqrt(np.mean((vel_true[1:] - vel_est[1:])**2))
print(f"Position RMSE: {rmse_pos:.4f}")
print(f"Velocity RMSE: {rmse_vel:.4f}")
print(f"Final velocity estimate: {vel_est[-1]:.4f}  (true = {vel_true[-1]:.4f})")
print(f"Final velocity 1σ:       {vel_std[-1]:.4f}")