import numpy as np

x = np.array([1, 2, 3])
y = np.array([40, 42, 45])

A = np.vstack([x, np.ones(len(x))]).T
a, b = np.linalg.lstsq(A, y, rcond=None)[0]

print(f"a={a} b={b}")