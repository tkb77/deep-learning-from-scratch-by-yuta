import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

fig = plt.figure()

target = 10
x = np.arange(0.0, 20.0, 0.1)
a = numerical_diff(function_1, target)

y1 = function_1(x)
y2 = a*(x - target) + function_1(target)

plt.plot(x, y1)
plt.plot(x, y2, linestyle="--")

plt.xlabel("x")
plt.ylabel("y")

fig.savefig("img.png")
