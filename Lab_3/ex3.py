import numpy as np

np.random.seed(42)

# --- Simulation ---
N = 50000
X = np.random.uniform(0, 1, N)

# --- Computation ---
sample_mean = np.mean(X)
theoretical_mean = 0.5
error = abs(sample_mean - theoretical_mean)

print(f"Sample Mean: {sample_mean:.5f}")
print(f"Theoretical Mean: {theoretical_mean:.5f}")
print(f"Difference: {error:.5f}")

# The computed empirical mean closely matches the theoretical expectation of 0.5. 
# This illustrates the Law of Large Numbers: the mathematical expectation represents 
# the long-term average that a stochastic process converges to over many iterations. 
# Any marginal discrepancy is purely due to finite sampling.
