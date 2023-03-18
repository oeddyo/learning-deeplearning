import numpy as np

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1

# propagate to first hidden layer
print(A1)
Z1 = 1 / (1 + np.exp(-A1))
print(Z1)

# propagate to second hidden layer
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = Z1 @ W2 + B2
Z2 = 1 / (1 + np.exp(-A2))

print(Z2)


# propagate to final output layer
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = Z2 @ W3 + B3
Y = A3

print(Y)
