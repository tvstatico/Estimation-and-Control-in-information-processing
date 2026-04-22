import numpy as np

np.random.seed(42)
N = 1000

# --- Simulation ---
true_x = 10
noise = np.random.normal(0, 2, N)
y = true_x + noise

# --- Estimation ---
x_hat = np.mean(y)
estimation_error = abs(true_x - x_hat)

print(f"True Value: {true_x}")
print(f"Estimated Value (Average): {x_hat:.4f}")
print(f"Estimation Error: {estimation_error:.4f}")

# Taking the simple arithmetic mean of the noisy observations yields an estimate 
# remarkably close to the true state. This proves that averaging is an effective 
# estimator capable of neutralizing zero-mean random noise. The residual error 
# is minor and scales down as the number of measurements increases.