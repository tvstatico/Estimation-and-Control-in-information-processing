import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Simulation ---
# Using a larger sample size to clearly show the true distribution shape
N = 50000
X = np.random.uniform(0, 1, N)

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.hist(X, bins=50, density=True, color='tab:blue', edgecolor='black', alpha=0.7)
plt.axhline(y=1, color='red', linestyle='--', label='Theoretical PDF')
plt.title("Ex2: Uniform Distribution (0, 1)")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# With a larger dataset, the histogram becomes significantly flatter and smoother, 
# perfectly approximating the theoretical Uniform distribution shape. This confirms 
# that every sub-interval within [0, 1] holds an equal probability of occurrence.