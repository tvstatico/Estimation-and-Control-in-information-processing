import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 10000

# --- Simulation & Visualization ---
variances = [0.5, 1.0, 3.0]
colors = ['tab:blue', 'tab:green', 'tab:red']

plt.figure(figsize=(10, 6))

for var, color in zip(variances, colors):
    std = np.sqrt(var)
    noise = np.random.normal(0, std, N)
    plt.hist(noise, bins=50, density=True, histtype='step', linewidth=2.5, 
             color=color, label=f'Variance = {var}')

plt.title("Ex6: Gaussian Noise with Different Variances")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# While all three zero-mean distributions share the same center, their widths vary 
# drastically. Increasing the variance flattens and widens the probability curve. 
# In a practical context, this translates to greater measurement uncertainty and 
# decreased system predictability.