import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 10000

# --- Simulation ---
X_gaussian = np.random.normal(0, 1, N)
X_uniform = np.random.uniform(-3, 3, N)

# --- Visualization ---
plt.figure(figsize=(10, 5))
plt.hist(X_gaussian, bins=50, alpha=0.6, color='tab:orange', density=True, label='Gaussian (mu=0, std=1)')
plt.hist(X_uniform, bins=50, alpha=0.4, color='tab:blue', density=True, label='Uniform (-3, 3)')

plt.title("Ex5: Gaussian vs. Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# Unlike the uniform distribution, which demonstrates equal probability across its range, 
# the Gaussian distribution forms a classic bell curve. This shape highlights that values 
# clustered near the mean are highly probable, whereas extreme outliers become 
# exponentially rarer.