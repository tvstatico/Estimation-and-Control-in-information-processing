import numpy as np

errors = np.array([-1, 2, -2])

squared_error = np.sum(errors**2)
absolute_error = np.sum(np.abs(errors))

print(f"Squared error: {squared_error}\nAbsolute error: {absolute_error}")

print("Squared error penalizes large errors more (sensitive to outliers)." if squared_error > absolute_error else "Absolute error is more robust to noise.")