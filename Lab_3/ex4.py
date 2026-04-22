import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 10000

# --- Simulation ---
# X1: Uniform (variance ~ 0.0833)
X1 = np.random.uniform(0, 1, N)
# X2: Gaussian with small variance (std = sqrt(0.01) = 0.1)
X2 = np.random.normal(0.5, 0.1, N)
# X3: Gaussian with large variance (std = sqrt(0.087) = 0.295)
X3 = np.random.normal(0.5, 0.295, N)

print(f"Variance of X1 (Uniform): {np.var(X1):.4f}")
print(f"Variance of X2 (Small):   {np.var(X2):.4f}")
print(f"Variance of X3 (Large):   {np.var(X3):.4f}")

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.hist(X1, bins=50, alpha=0.4, label='Uniform (Moderate spread)', density=True)
plt.hist(X2, bins=50, alpha=0.6, label='Small Variance (Tight cluster)', density=True)
plt.hist(X3, bins=50, alpha=0.5, label='Large Variance (High dispersion)', density=True)

plt.title("Ex4: Comparison of Variances")
plt.legend()
plt.show()

# Variance quantifies dispersion. The uniform distribution establishes a baseline spread. 
# A dataset with low variance (X2) is tightly clustered around its mean, representing 
# high certainty. Conversely, high variance (X3) implies a wider spread of values 
# across the domain, signifying higher statistical dispersion.