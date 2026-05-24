# Exercise 7. Effect of the measurement noise covariance R
# Repeat the Kalman filter simulation for:
#     R=0.01
#     R=0.5
#     R=5
# Plot the estimated state for each case and compare the results.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- True system parameters (fixed) ----
A = 1.0
B = 1.0
C = 1.0

N = 30
u = 1.0

# True noise levels for DATA GENERATION (fixed across all runs)
sigma_w_true = 0.1              # true process noise std  -> true Q = 0.01
sigma_v_true = np.sqrt(0.5)     # true measurement noise std -> true R = 0.5

Q = 0.01                        # filter's Q (kept correct)

# ---- Generate ONE ground truth and ONE measurement sequence ----
x_true = np.zeros(N+1)
y_meas = np.zeros(N+1)

x_true[0] = 0.0
y_meas[0] = C * x_true[0] + sigma_v_true * np.random.randn()

for t in range(N):
    w = sigma_w_true * np.random.randn()
    v = sigma_v_true * np.random.randn()
    x_true[t+1] = A * x_true[t] + B * u + w
    y_meas[t+1] = C * x_true[t+1] + v


# ---- Kalman filter as a function of R ----
def run_kalman(R_filter, y, x0=0.0, P0=1.0):
    x_est = np.zeros(N+1)
    P_est = np.zeros(N+1)
    K_all = np.zeros(N+1)

    x_est[0] = x0
    P_est[0] = P0

    for t in range(N):
        # Prediction
        x_pred = A * x_est[t] + B * u
        P_pred = A * P_est[t] * A + Q
        # Update
        S = C * P_pred * C + R_filter
        K = P_pred * C / S
        x_est[t+1] = x_pred + K * (y[t+1] - C * x_pred)
        P_est[t+1] = (1 - K * C) * P_pred
        K_all[t+1] = K

    return x_est, P_est, K_all


# ---- Run for three R values ----
R_values = [0.01, 0.5, 5.0]
colors   = ['tab:blue', 'tab:green', 'tab:red']
results  = {}

for R in R_values:
    x_est, P_est, K_all = run_kalman(R, y_meas)
    mse = np.mean((x_true[1:] - x_est[1:])**2)
    results[R] = {'x_est': x_est, 'P': P_est, 'K': K_all, 'MSE': mse}

# ---- Print MSE table ----
print("===== Effect of R on Kalman filter =====")
print(f"{'R':>10} | {'MSE':>8} | {'K_ss':>8} | {'P_ss':>8}")
print("-" * 42)
for R in R_values:
    r = results[R]
    print(f"{R:>10.4f} | {r['MSE']:>8.4f} | {r['K'][-1]:>8.4f} | {r['P'][-1]:>8.4f}")

time = np.arange(N+1)

# ======================================================================
# Plot 1: estimates for all three R values, vs truth and measurements
# ======================================================================
plt.figure(figsize=(10, 5.5))
plt.plot(time, x_true, 'k-o', label='True state',
         linewidth=2.5, markersize=5)
plt.plot(time, y_meas, 'x', color='gray', label='Measurements',
         markersize=7, alpha=0.6)

for R, col in zip(R_values, colors):
    r = results[R]
    plt.plot(time, r['x_est'], '-', color=col, linewidth=2,
             label=f'Kalman, R={R}  (MSE={r["MSE"]:.3f})')

plt.xlabel('Time step t')
plt.ylabel('Value')
plt.title('Kalman Estimate for Different Values of R')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ======================================================================
# Plot 2: Kalman gain and covariance for all three R
# ======================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for R, col in zip(R_values, colors):
    r = results[R]
    ax1.plot(time[1:], r['K'][1:], '-o', color=col, markersize=4,
             label=f'R={R}')
    ax2.plot(time, r['P'], '-o', color=col, markersize=4,
             label=f'R={R}')

ax1.set_ylabel('Kalman gain K(t)')
ax1.set_title('Kalman Gain and Posterior Covariance for Different R')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Time step t')
ax2.set_ylabel('Covariance P(t|t)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()