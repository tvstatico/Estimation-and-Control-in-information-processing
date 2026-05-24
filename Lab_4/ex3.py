import numpy as np

x = np.array([1, 2, 3])
y = np.array([40, 42, 45])

a = 2.5
b = 37.33333333333334

y_pred = a * x + b
residuals = y - y_pred

print(f"Predicted values: {y_pred}")
print(f"Residuals: {residuals}")
print(f"Squared error: {np.sum(residuals**2)}")