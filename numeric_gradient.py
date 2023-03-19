import numpy as np


def numeric_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        t1 = x[i] - h
        t2 = x[i] + h

        v1 = f(t1)
        v2 = f(t2)

        grad[i] = (v2 - v1) / (2*h)

    return grad


def function_2(x):
    return np.sum(x**2)


def gradient_descent(f, init_x, lr=0.01, step_num=1000):
    x = init_x
    for i in range(step_num):
        grad = numeric_gradient(f, x)
        x -= lr * grad
    return x


print(numeric_gradient(function_2, np.array([3.0, 4.0])))

print(gradient_descent(function_2, np.array([100.0, 100.0])))


