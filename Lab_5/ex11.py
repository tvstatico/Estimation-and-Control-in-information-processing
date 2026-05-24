# Exercise 11. Nonlinear system
# Consider the nonlinear system:
# x(t+1)=x(t)+0.1*x^2(t)+u(t)
# Simulate the system and write a short explanation of why the standard Kalman
# filter is not exact for this model. Then explain, in words, the main idea of
# the Extended Kalman Filter.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ---- Simulation setup ----
N = 20                        # short horizon; system is explosive
u_const = 0.0                 # zero control input
x0 = 0.5                      # initial state (makes the x^2 term matter)

Q = 0.0001                    # small process noise
R = 0.05                      # measurement noise
sigma_w = np.sqrt(Q)
sigma_v = np.sqrt(R)

def f(x, u):                  # nonlinear dynamics
    return x + 0.1 * x**2 + u

# ---- Simulate true nonlinear trajectory + measurements ----
x_true = np.zeros(N+1)
y_meas = np.zeros(N+1)

x_true[0] = x0
y_meas[0] = x_true[0] + sigma_v * np.random.randn()

for t in range(N):
    w = sigma_w * np.random.randn()
    v = sigma_v * np.random.randn()
    x_true[t+1] = f(x_true[t], u_const) + w
    y_meas[t+1] = x_true[t+1] + v

# ---- Run a STANDARD Kalman filter that wrongly assumes A = 1 ----
# This ignores the 0.1*x^2 term entirely in the prediction model.
A_linear = 1.0
B_linear = 1.0
C_linear = 1.0

x_kf = np.zeros(N+1)
P_kf = np.zeros(N+1)
x_kf[0] = x0
P_kf[0] = 1.0

for t in range(N):
    # Prediction (using WRONG linear model)
    x_pred = A_linear * x_kf[t] + B_linear * u_const
    P_pred = A_linear * P_kf[t] * A_linear + Q
    # Update
    S = C_linear * P_pred * C_linear + R
    K = P_pred * C_linear / S
    x_kf[t+1] = x_pred + K * (y_meas[t+1] - C_linear * x_pred)
    P_kf[t+1] = (1 - K * C_linear) * P_pred

# ---- Also run an EKF for comparison (optional but illuminating) ----
# The EKF uses the actual nonlinear f for prediction and linearizes
# only to propagate the covariance.
def F_jacobian(x):
    # df/dx = 1 + 0.2 * x
    return 1.0 + 0.2 * x

x_ekf = np.zeros(N+1)
P_ekf = np.zeros(N+1)
x_ekf[0] = x0
P_ekf[0] = 1.0

for t in range(N):
    # Prediction with the TRUE nonlinear dynamics
    x_pred = f(x_ekf[t], u_const)
    F_t    = F_jacobian(x_ekf[t])     # linearization at the current estimate
    P_pred = F_t * P_ekf[t] * F_t + Q
    # Standard update (C is already linear here)
    S = P_pred + R
    K = P_pred / S
    x_ekf[t+1] = x_pred + K * (y_meas[t+1] - x_pred)
    P_ekf[t+1] = (1 - K) * P_pred

time = np.arange(N+1)

# ======================================================================
# Plot: true state, measurements, linear KF estimate, EKF estimate
# ======================================================================
plt.figure(figsize=(10, 6))
plt.plot(time, x_true, 'b-o',  label='True state (nonlinear)',
         linewidth=2, markersize=5)
plt.plot(time, y_meas, 'kx',   label='Measurements',
         markersize=7, alpha=0.6)
plt.plot(time, x_kf,   'r-s',  label='Standard KF (assumes A=1)',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.plot(time, x_ekf,  'g-d',  label='EKF (linearizes at each step)',
         linewidth=2, markersize=5, markerfacecolor='none')
plt.xlabel('Time step t')
plt.ylabel('State x(t)')
plt.title('Nonlinear System: Standard KF vs. Extended KF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Diagnostics ----
rmse_kf  = np.sqrt(np.mean((x_true[1:] - x_kf[1:])**2))
rmse_ekf = np.sqrt(np.mean((x_true[1:] - x_ekf[1:])**2))
print(f"Standard KF RMSE: {rmse_kf:.4f}")
print(f"EKF         RMSE: {rmse_ekf:.4f}")
print(f"Final x_true = {x_true[-1]:.3f}")
print(f"Final x_KF   = {x_kf[-1]:.3f}")
print(f"Final x_EKF  = {x_ekf[-1]:.3f}")