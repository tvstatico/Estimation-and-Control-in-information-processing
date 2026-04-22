import matplotlib.pyplot as plt

# --- Configuration ---
r_array = [2, 2.5, 1, 1.2, 3.1, 0.5, 4, 4.4, 3, 2.9, 2.8, 1.9, 1.5, 1.4, 7, 3.8, 8]
x0 = 0.5
n_steps = 20

# --- Simulation & Visualization ---
plt.figure(figsize=(12, 8))

for r in r_array:
    history = [x0]
    x = x0
    diverged = False
    
    for _ in range(n_steps):
        try:
            x = r * x * (1 - x)
            # Catch massive divergence caused by r > 4
            if x < -1000 or x > 1000:
                diverged = True
                break
            history.append(x)
        except OverflowError:
            diverged = True
            break
            
    if not diverged:
        plt.plot(history, alpha=0.6, label=f"r={r}")
    else:
        # Plot only the steps before it diverged
        plt.plot(history, linestyle=':', linewidth=2, label=f"r={r} (Diverged)")

plt.title("Ex3: Population Size for Various r Values")
plt.xlabel("Generation")
plt.ylabel("Population (x)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()