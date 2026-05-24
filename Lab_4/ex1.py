import numpy as np

temperatures = np.array([21, 22, 23, 24])

calibrated_temp = np.mean(temperatures)
errors = temperatures - calibrated_temp
mse = np.mean(errors**2)

print(f"Calibrated temperature: {calibrated_temp}")
print(f"Errors: {errors}")
print(f"MSE: {mse}")