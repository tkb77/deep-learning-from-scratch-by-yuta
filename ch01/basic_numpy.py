import numpy as np

x = np.array([1.0,2.0,3.0])
y = np.array([2.0,4.0,6.0])
z = 2

print(x + y)
print(x / z)

A = np.array([[1,2],[3,4]])
B = np.array([2.0,2.0])
print(A * B)
print(A[1][1])

X = np.array([[51,55],[14,19],[0,4]])
X = X.flatten()

print(X[np.array([0,5])])
