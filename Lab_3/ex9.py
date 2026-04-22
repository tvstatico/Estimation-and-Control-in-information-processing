import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 1000

# --- Simulation ---
# Generate X
X = np.random.normal(0, 1, N)
# Generate Y directly dependent on X plus some noise
Y = 2 * X + np.random.normal(0, 1, N)

# --- Computation ---
cov_matrix = np.cov(X, Y)
covariance = cov_matrix[0, 1]
print(f"Calculated Covariance: {covariance:.4f}")

# --- Visualization ---
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, alpha=0.5, color='tab:brown', edgecolors='k')
plt.title(f"Ex9: Correlated Variables (Covariance ≈ {covariance:.2f})")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.grid(True, linestyle='--')
plt.show()

# The positive covariance value confirms a strong direct relationship: as X increases, 
# Y proportionally increases. The distinct linear upward trend in the scatter plot 
# visually corroborates this strong positive correlation.