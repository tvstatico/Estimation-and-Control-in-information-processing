import matplotlib.pyplot as plt

# --- Configuration ---
r = 0.9
x0 = 0.5  # Starting population (50% of maximum capacity)
n_steps = 20

# --- Simulation ---
history = [x0]
x = x0

for _ in range(n_steps):
    x = r * x * (1 - x)
    history.append(x)

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.plot(history, marker='o', color='tab:blue')
plt.title(f"Ex1: Population Growth (r = {r})")
plt.xlabel("Generation")
plt.ylabel("Population Size (x)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
