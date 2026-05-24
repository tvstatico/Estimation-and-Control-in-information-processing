# Exercise 8. Covariance analysis
# Plot both P(t|t-1) and P(t|t) on the same graph.
# Write a short explanation of the difference between predicted covariance and
# updated covariance.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- System and filter parameters ----
A = 1.0
B = 1.0
C = 1.0

N = 30
u = 1.0

Q = 0.01
R = 0.5

sigma_w = np.sqrt(Q)
sigma_v = np.sqrt(R)

# ---- Allocate arrays ----
x_true  = np.zeros(N+1)
y_meas  = np.zeros(N+1)
x_est   = np.zeros(N+1)       # x_hat(t|t)
P_post  = np.zeros(N+1)       # P(t|t)      - posterior (updated) covariance
P_prior = np.full(N+1, np.nan)  # P(t|t-1)  - prior (predicted) covariance
K_all   = np.zeros(N+1)

# ---- Initial conditions ----
x_true[0]  = 0.0
x_est[0]   = 0.0
P_post[0]  = 1.0              # P(0|0)
# P(0|-1) is undefined -> leave as NaN

y_meas[0] = C * x_true[0] + sigma_v * np.random.randn()

# ---- Simulate + Kalman filter, logging both covariances ----
for t in range(N):
    # True system
    w = sigma_w * np.random.randn()
    v = sigma_v * np.random.randn()
    x_true[t+1] = A * x_true[t] + B * u + w
    y_meas[t+1] = C * x_true[t+1] + v

    # Prediction
    x_pred = A * x_est[t] + B * u
    P_pred = A * P_post[t] * A + Q
    P_prior[t+1] = P_pred       # <-- log the prior covariance

    # Update
    S = C * P_pred * C + R
    K = P_pred * C / S
    x_est[t+1]  = x_pred + K * (y_meas[t+1] - C * x_pred)
    P_post[t+1] = (1 - K * C) * P_pred
    K_all[t+1]  = K

time = np.arange(N+1)

# ======================================================================
# Plot: P(t|t-1) and P(t|t) overlaid
# ======================================================================
plt.figure(figsize=(10, 5.5))

# Plot prior covariance (starts at t=1)
plt.plot(time[1:], P_prior[1:], 'r-^', linewidth=2, markersize=7,
         label='Predicted covariance  P(t|t-1)  — prior')

# Plot posterior covariance
plt.plot(time, P_post, 'g-o', linewidth=2, markersize=6,
         label='Updated covariance   P(t|t)    — posterior')

# Connect each prior-posterior pair with a thin dashed line to
# visualise the "drop" at each update step
for t in range(1, N+1):
    plt.plot([t, t], [P_prior[t], P_post[t]],
             'k:', linewidth=0.8, alpha=0.5)

plt.xlabel('Time step t')
plt.ylabel('Covariance')
plt.title('Predicted vs. Updated Covariance in the Kalman Filter')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Diagnostics ----
print(f"Steady-state prior     P(t|t-1) ≈ {P_prior[-1]:.4f}")
print(f"Steady-state posterior P(t|t)   ≈ {P_post[-1]:.4f}")
print(f"Reduction per update           ≈ {P_prior[-1] - P_post[-1]:.4f}")
print(f"Relative reduction              ≈ {100*(1 - P_post[-1]/P_prior[-1]):.1f}%")