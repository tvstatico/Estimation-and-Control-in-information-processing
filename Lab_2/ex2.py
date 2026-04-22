import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
r_values = np.linspace(2.5, 4.0, 10000)
iterations = 1000
last_iterations = 100

# --- Simulation ---
x = 1e-5 * np.ones_like(r_values)

plt.figure(figsize=(10, 6))
plt.title("Ex2: Bifurcation Diagram (Onset of Chaos)")

for i in range(iterations):
    x = r_values * x * (1 - x)
    # Only plot the steady states or chaotic attractors
    if i >= (iterations - last_iterations):
        plt.plot(r_values, x, ',k', alpha=0.05)

# Highlight the onset of chaos
chaos_onset = 3.56995
plt.axvline(x=chaos_onset, color='red', linestyle='--', label=f'Onset of Chaos (r ≈ {chaos_onset})')

plt.xlabel("Growth factor (r)")
plt.ylabel("Population (x)")
plt.legend()
plt.tight_layout()
plt.show()
