import numpy as np
import matplotlib.pyplot as plt

time = np.array([1, 2, 3, 4, 5])
energy = np.array([10, 12, 15, 18, 20])

a, b = np.polyfit(time, energy, 1)

print(f"a (consumption rate) = {a}")
print(f"b = {b}")

energy_pred = a * time + b

plt.scatter(time, energy, label="Data")
plt.plot(time, energy_pred, color='red', label="Fitted line")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.legend()
plt.show()