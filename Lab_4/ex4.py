import numpy as np

sensor1 = np.array([1, 2])
sensor2 = np.array([2, 4])

A = np.column_stack((sensor1, sensor2))
rank = np.linalg.matrix_rank(A)

print(f"A =\n{A}")
print(f"Rank = {rank}")

if rank < A.shape[1]:
    print("Columns are linearly dependent\nSystem fails because A^T A is singular")
else:
    print("Columns are independent")