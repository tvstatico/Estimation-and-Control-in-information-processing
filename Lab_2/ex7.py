import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Configuration ---
states = ['a', 'b', 'c']
# Modify weight for 'b' to 0.5
weights = {'a': 1.5, 'b': 0.5, 'c': 3.3} 
n_steps = 15
x = 0.5
history = [x]

# Re-generate the exact same sequence of states
state_sequence = np.random.choice(states, size=n_steps)

# --- Simulation ---
for state in state_sequence:
    r = weights[state]
    x = r * x * (1 - x)
    history.append(x)

# --- Visualization ---
plt.figure(figsize=(8, 4))
plt.plot(range(n_steps + 1), history, marker='s', color='brown')
plt.title("Ex7: Modified Markov Machine (b=0.5)")
plt.xlabel("Step")
plt.ylabel("Population")
plt.xticks(range(n_steps + 1))
plt.grid(True, linestyle=':')
plt.show()

print(f"Final population after 15 steps: {history[-1]:.6f}")
