import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Simulation ---
# Generate 1000 random numbers from the standard uniform distribution
data = np.random.rand(1000)

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.hist(data, bins=20, color='tab:cyan', edgecolor='black', alpha=0.7)
plt.title("Ex1: Histogram of 1000 Random Numbers")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# The resulting histogram is roughly flat across the [0, 1] interval, indicating 
# an even distribution of generated values. While the bars are not perfectly level, 
# these minor fluctuations are a natural consequence of the finite sample size (1000) 
# and the inherent randomness of the generator.