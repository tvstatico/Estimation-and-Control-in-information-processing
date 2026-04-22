import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Configuration ---
states = ['a', 'b', 'c']
weights = {'a': 1.5, 'b': 2.0, 'c': 3.3}
n_steps = 15
x = 0.5
history = [x]

# Generate a sequence of 15 states randomly
state_sequence = np.random.choice(states, size=n_steps)

print("Ex6: Sequence and Growth")
print(f"Initial population: {x}")

# --- Simulation ---
for step, state in enumerate(state_sequence):
    r = weights[state]
    x = r * x * (1 - x)
    history.append(x)
    print(f"Step {step+1}: State = {state}, r = {r}, Population = {x:.4f}")

# --- Visualization ---
plt.figure(figsize=(8, 4))
plt.plot(range(n_steps + 1), history, marker='s', color='purple')
plt.title("Ex6: Markov Machine Driven Growth")
plt.xlabel("Step")
plt.ylabel("Population")
plt.xticks(range(n_steps + 1))
plt.grid(True, linestyle=':')
plt.show()