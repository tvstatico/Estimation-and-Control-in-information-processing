import matplotlib.pyplot as plt

# --- Configuration ---
r = 4
initial_populations = [0.1, 0.5, 1.0, 2.0]
n_steps = 15

plt.figure(figsize=(10, 6))

# --- Simulation ---
for x0 in initial_populations:
    x = x0
    history = [x]
    
    for _ in range(n_steps):
        x = r * x * (1 - x)
        history.append(x)
        
    plt.plot(history, marker='o', label=f"Start: {x0}")

plt.title("Ex5: Extreme Initial Conditions (r = 4)")
plt.xlabel("Generation")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()
