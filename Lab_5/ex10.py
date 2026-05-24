# Exercise 10. State feedback from estimate
# Use the control law: u(t)=-Lx_hat(t|t)
# Choose L=0.5.
# Use the estimated state in the control law and simulate the closed loop system.
# Plot the state and the control signal.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- System parameters ----
A = 1.0
B = 1.0
C = 1.0

N = 30

# Feedback gain
L = 0.5                      # closed-loop pole = A - B*L = 0.5  (stable)

# Noise and filter tuning
Q = 0.01
R = 0.5
sigma_w = np.sqrt(Q)
sigma_v = np.sqrt(R)

# ---- Allocate arrays ----
x_true = np.zeros(N+1)
y_meas = np.zeros(N+1)
x_est  = np.zeros(N+1)
P      = np.zeros(N+1)
u_hist = np.zeros(N+1)       # u(t) applied at time t

# ---- Initial conditions ----
# Start the plant away from zero so the regulator has something to do
x_true[0] = 10.0
x_est[0]  = 0.0              # filter starts with a wrong initial guess
P[0]      = 10.0             # but large initial uncertainty reflects that

y_meas[0] = C * x_true[0] + sigma_v * np.random.randn()

# Initial control action, using the initial estimate
u_hist[0] = -L * x_est[0]

# ---- Closed-loop simulation ----
for t in range(N):
    # --- Control: u(t) based on the CURRENT estimate ---
    u = -L * x_est[t]
    u_hist[t] = u

    # --- True plant evolves using this u ---
    w = sigma_w * np.random.randn()
    v = sigma_v * np.random.randn()
    x_true[t+1] = A * x_true[t] + B * u + w
    y_meas[t+1] = C * x_true[t+1] + v

    # --- Kalman prediction (must use the SAME u the plant saw) ---
    x_pred = A * x_est[t] + B * u
    P_pred = A * P[t] * A + Q

    # --- Kalman update ---
    S = C * P_pred * C + R
    K = P_pred * C / S
    x_est[t+1] = x_pred + K * (y_meas[t+1] - C * x_pred)
    P[t+1]     = (1 - K * C) * P_pred

# Final control step (for plotting continuity)
u_hist[N] = -L * x_est[N]

time = np.arange(N+1)

# ======================================================================
# Plot 1: true state, estimated state, measurements
# ======================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(time, x_true, 'b-o', label='True state x(t)',
         linewidth=2, markersize=5)
ax1.plot(time, x_est,  'g-s', label='Estimated state x̂(t|t)',
         linewidth=2, markersize=5, markerfacecolor='none')
ax1.plot(time, y_meas, 'rx',  label='Measurement y(t)',
         markersize=7, alpha=0.6)
ax1.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5,
            label='Setpoint = 0')
ax1.set_ylabel('State')
ax1.set_title('Closed-Loop Response with State-Feedback Control  u(t) = -L·x̂(t|t)')
ax1.legend(loc='upper right')
ax1.grid(True)

# ======================================================================
# Plot 2: control signal
# ======================================================================
ax2.plot(time, u_hist, 'm-o', linewidth=2, markersize=5,
         label='Control signal u(t) = -L·x̂(t|t)')
ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Time step t')
ax2.set_ylabel('Control input u(t)')
ax2.set_title('Control Signal over Time')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# ---- Diagnostics ----
print(f"Closed-loop pole (deterministic):  A - B*L = {A - B*L}")
print(f"Final true state      x({N})   = {x_true[-1]:.4f}")
print(f"Final estimated state x̂({N}|{N}) = {x_est[-1]:.4f}")
print(f"Final control input   u({N})   = {u_hist[-1]:.4f}")
print(f"Settling behavior: state should decay toward 0 as (A-BL)^t = 0.5^t")